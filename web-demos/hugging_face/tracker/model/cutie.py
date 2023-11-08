from typing import List, Dict
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn

from tracker.model.modules import *
from tracker.model.big_modules import *
from tracker.model.aux_modules import AuxComputer
from tracker.model.utils.memory_utils import *
from tracker.model.transformer.object_transformer import QueryTransformer
from tracker.model.transformer.object_summarizer import ObjectSummarizer
from tracker.utils.tensor_utils import aggregate

log = logging.getLogger()


class CUTIE(nn.Module):
    def __init__(self, cfg: DictConfig, *, single_object=False):
        super().__init__()
        model_cfg = cfg.model
        self.ms_dims = model_cfg.pixel_encoder.ms_dims
        self.key_dim = model_cfg.key_dim
        self.value_dim = model_cfg.value_dim
        self.sensory_dim = model_cfg.sensory_dim
        self.pixel_dim = model_cfg.pixel_dim
        self.embed_dim = model_cfg.embed_dim
        self.single_object = single_object

        log.info(f'Single object: {self.single_object}')

        self.pixel_encoder = PixelEncoder(model_cfg)
        self.pix_feat_proj = nn.Conv2d(self.ms_dims[0], self.pixel_dim, kernel_size=1)
        self.key_proj = KeyProjection(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg, single_object=single_object)
        self.mask_decoder = MaskDecoder(model_cfg)
        self.pixel_fuser = PixelFeatureFuser(model_cfg, single_object=single_object)
        self.object_transformer = QueryTransformer(model_cfg)
        self.object_summarizer = ObjectSummarizer(model_cfg)
        self.aux_computer = AuxComputer(cfg)

        self.register_buffer("pixel_mean", torch.Tensor(model_cfg.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(model_cfg.pixel_std).view(-1, 1, 1), False)

    def _get_others(self, masks: torch.Tensor) -> torch.Tensor:
        # for each object, return the sum of masks of all other objects
        if self.single_object:
            return None

        num_objects = masks.shape[1]
        if num_objects >= 1:
            others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)
        else:
            others = torch.zeros_like(masks)
        return others

    def encode_image(self, image: torch.Tensor) -> (Iterable[torch.Tensor], torch.Tensor):
        image = (image - self.pixel_mean) / self.pixel_std
        ms_image_feat = self.pixel_encoder(image)
        return ms_image_feat, self.pix_feat_proj(ms_image_feat[0])

    def encode_mask(
            self,
            image: torch.Tensor,
            ms_features: List[torch.Tensor],
            sensory: torch.Tensor,
            masks: torch.Tensor,
            *,
            deep_update: bool = True,
            chunk_size: int = -1,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image = (image - self.pixel_mean) / self.pixel_std
        others = self._get_others(masks)
        mask_value, new_sensory = self.mask_encoder(image,
                                                    ms_features,
                                                    sensory,
                                                    masks,
                                                    others,
                                                    deep_update=deep_update,
                                                    chunk_size=chunk_size)
        object_summaries, object_logits = self.object_summarizer(masks, mask_value, need_weights)
        return mask_value, new_sensory, object_summaries, object_logits

    def transform_key(self,
                      final_pix_feat: torch.Tensor,
                      *,
                      need_sk: bool = True,
                      need_ek: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        key, shrinkage, selection = self.key_proj(final_pix_feat, need_s=need_sk, need_e=need_ek)
        return key, shrinkage, selection

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key: torch.Tensor, query_selection: torch.Tensor,
                    memory_key: torch.Tensor, memory_shrinkage: torch.Tensor,
                    msk_value: torch.Tensor, obj_memory: torch.Tensor, pix_feat: torch.Tensor,
                    sensory: torch.Tensor, last_mask: torch.Tensor,
                    selector: torch.Tensor) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        msk_value       : B * num_objects * CV * T * H * W
        obj_memory      : B * num_objects * T * num_summaries * C
        pixel_feature   : B * C * H * W
        """
        batch_size, num_objects = msk_value.shape[:2]

        # read using visual attention
        with torch.cuda.amp.autocast(enabled=False):
            affinity = get_affinity(memory_key.float(), memory_shrinkage.float(), query_key.float(),
                                    query_selection.float())

            msk_value = msk_value.flatten(start_dim=1, end_dim=2).float()

            # B * (num_objects*CV) * H * W
            pixel_readout = readout(affinity, msk_value)
            pixel_readout = pixel_readout.view(batch_size, num_objects, self.value_dim,
                                               *pixel_readout.shape[-2:])
        pixel_readout = self.pixel_fusion(pix_feat, pixel_readout, sensory, last_mask)

        # read from query transformer
        mem_readout, aux_features = self.readout_query(pixel_readout, obj_memory, selector=selector)

        aux_output = {
            'sensory': sensory,
            'q_logits': aux_features['logits'] if aux_features else None,
            'attn_mask': aux_features['attn_mask'] if aux_features else None,
        }

        return mem_readout, aux_output

    def pixel_fusion(self,
                     pix_feat: torch.Tensor,
                     pixel: torch.Tensor,
                     sensory: torch.Tensor,
                     last_mask: torch.Tensor,
                     *,
                     chunk_size: int = -1) -> torch.Tensor:
        last_mask = F.interpolate(last_mask, size=sensory.shape[-2:], mode='area')
        last_others = self._get_others(last_mask)
        fused = self.pixel_fuser(pix_feat,
                                 pixel,
                                 sensory,
                                 last_mask,
                                 last_others,
                                 chunk_size=chunk_size)
        return fused

    def readout_query(self,
                      pixel_readout,
                      obj_memory,
                      *,
                      selector=None,
                      need_weights=False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        return self.object_transformer(pixel_readout,
                                       obj_memory,
                                       selector=selector,
                                       need_weights=need_weights)

    def segment(self,
                ms_image_feat: List[torch.Tensor],
                memory_readout: torch.Tensor,
                sensory: torch.Tensor,
                *,
                selector: bool = None,
                chunk_size: int = -1,
                update_sensory: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        multi_scale_features is from the key encoder for skip-connection
        memory_readout is from working/long-term memory
        sensory is the sensory memory
        last_mask is the mask from the last frame, supplementing sensory memory
        selector is 1 if an object exists, and 0 otherwise. We use it to filter padded objects
            during training.
        """
        sensory, logits = self.mask_decoder(ms_image_feat,
                                            memory_readout,
                                            sensory,
                                            chunk_size=chunk_size,
                                            update_sensory=update_sensory)

        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        # Softmax over all objects[]
        logits = aggregate(prob, dim=1)
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        prob = F.softmax(logits, dim=1)

        return sensory, logits, prob

    def compute_aux(self, pix_feat: torch.Tensor, aux_inputs: Dict[str, torch.Tensor],
                    selector: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.aux_computer(pix_feat, aux_inputs, selector)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        if not self.single_object:
            # Map single-object weight to multi-object weight (4->5 out channels in conv1)
            for k in list(src_dict.keys()):
                if k == 'mask_encoder.conv1.weight':
                    if src_dict[k].shape[1] == 4:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
                elif k == 'pixel_fuser.sensory_compress.weight':
                    if src_dict[k].shape[1] == self.sensory_dim + 1:
                        log.info(f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
        elif self.single_object:
            """
            If the model is multiple-object and we are training in single-object, 
            we strip the last channel of conv1.
            This is not supposed to happen in standard training except when users are trying to
            finetune a trained model with single object datasets.
            """
            if src_dict['mask_encoder.conv1.weight'].shape[1] == 5:
                log.warning(f'Converting {k} from multiple objects to single object.'
                            'This is not supposed to happen in standard training.')
                src_dict[k] = src_dict[k][:, :-1]

        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')

        self.load_state_dict(src_dict, strict=False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device