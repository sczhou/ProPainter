from typing import List, Optional, Iterable, Dict
import logging
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F

from tracker.inference.memory_manager import MemoryManager
from tracker.inference.object_manager import ObjectManager
from tracker.inference.image_feature_store import ImageFeatureStore
from tracker.model.cutie import CUTIE
from tracker.utils.tensor_utils import pad_divide_by, unpad, aggregate

log = logging.getLogger()


class InferenceCore:
    def __init__(self,
                 network: CUTIE,
                 cfg: DictConfig,
                 *,
                 image_feature_store: ImageFeatureStore = None):
        self.network = network
        self.cfg = cfg
        self.mem_every = cfg.mem_every
        stagger_updates = cfg.stagger_updates
        self.chunk_size = cfg.chunk_size
        self.save_aux = cfg.save_aux
        self.max_internal_size = cfg.max_internal_size
        self.flip_aug = cfg.flip_aug

        self.curr_ti = -1
        self.last_mem_ti = 0
        # at which time indices should we update the sensory memory
        if stagger_updates >= self.mem_every:
            self.stagger_ti = set(range(1, self.mem_every + 1))
        else:
            self.stagger_ti = set(
                np.round(np.linspace(1, self.mem_every, stagger_updates)).astype(int))
        self.object_manager = ObjectManager()
        self.memory = MemoryManager(cfg=cfg, object_manager=self.object_manager)

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(self.network)
        else:
            self.image_feature_store = image_feature_store

        self.last_mask = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(cfg=self.cfg, object_manager=self.object_manager)

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_sensory_memory()

    def update_config(self, cfg):
        self.mem_every = cfg['mem_every']
        self.memory.update_config(cfg)

    def _add_memory(self,
                    image: torch.Tensor,
                    pix_feat: torch.Tensor,
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True,
                    force_permanent: bool = False) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used.
        image: RGB image, (1/2)*3*H*W
        pix_feat: from the key encoder, (1/2)*_*H*W
        prob: (1/2)*num_objects*H*W, in [0, 1]
        key/shrinkage/selection: for anisotropic l2, (1/2)*_*H*W
        selection can be None if not using long-term memory
        is_deep_update: whether to use deep update (e.g. with the mask encoder)
        force_permanent: whether to force the memory to be permanent
        """
        if prob.shape[1] == 0:
            # nothing to add
            log.warn('Trying to add an empty object mask to memory!')
            return

        if force_permanent:
            as_permanent = 'all'
        else:
            as_permanent = 'first'

        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)
        msk_value, sensory, obj_value, self.obj_logits = self.network.encode_mask(
            image,
            pix_feat,
            self.memory.get_sensory(self.object_manager.all_obj_ids),
            prob,
            deep_update=is_deep_update,
            chunk_size=self.chunk_size,
            need_weights=self.save_aux)
        self.memory.add_memory(key,
                               shrinkage,
                               msk_value,
                               obj_value,
                               self.object_manager.all_obj_ids,
                               selection=selection,
                               as_permanent=as_permanent)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 pix_feat: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        key/selection: for anisotropic l2: (1/2) * _ * H * W
        pix_feat: from the key encoder, (1/2) * _ * H * W
        ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        update_sensory: whether to update the sensory memory

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        bs = key.shape[0]
        if self.flip_aug:
            assert bs == 2
        else:
            assert bs == 1

        if not self.memory.engaged:
            log.warn('Trying to segment without any memory!')
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)

        memory_readout = self.memory.read(pix_feat, key, selection, self.last_mask, self.network)
        memory_readout = self.object_manager.realize_dict(memory_readout)
        sensory, _, pred_prob_with_bg = self.network.segment(ms_features,
                                                             memory_readout,
                                                             self.memory.get_sensory(
                                                                 self.object_manager.all_obj_ids),
                                                             chunk_size=self.chunk_size,
                                                             update_sensory=update_sensory)
        # remove batch dim
        if self.flip_aug:
            # average predictions of the non-flipped and flipped version
            pred_prob_with_bg = (pred_prob_with_bg[0] +
                                 torch.flip(pred_prob_with_bg[1], dims=[-1])) / 2
        else:
            pred_prob_with_bg = pred_prob_with_bg[0]
        if update_sensory:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)
        return pred_prob_with_bg

    def step(self,
             image: torch.Tensor,
             mask: Optional[torch.Tensor] = None,
             objects: Optional[List[int]] = None,
             *,
             idx_mask: bool = True,
             end: bool = False,
             delete_buffer: bool = True,
             force_permanent: bool = False) -> torch.Tensor:
        """
        Take a step with a new incoming image.
        If there is an incoming mask with new objects, we will memorize them.
        If there is no incoming mask, we will segment the image using the memory.
        In both cases, we will update the memory and return a segmentation.

        image: 3*H*W
        mask: H*W (if idx mask) or len(objects)*H*W or None
        objects: list of object ids that are valid in the mask Tensor.
                The ids themselves do not need to be consecutive/in order, but they need to be 
                in the same position in the list as the corresponding mask
                in the tensor in non-idx-mask mode.
                objects is ignored if the mask is None. 
                If idx_mask is False and objects is None, we sequentially infer the object ids.
        idx_mask: if True, mask is expected to contain an object id at every pixel.
                  If False, mask should have multiple channels with each channel representing one object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False 
        delete_buffer: whether to delete the image feature buffer after this step
        force_permanent: the memory recorded this frame will be added to the permanent memory
        """
        if objects is None and mask is not None:
            assert not idx_mask
            objects = list(range(1, mask.shape[0] + 1))

        # resize input if needed -- currently only used for the GUI
        resize_needed = False
        if self.max_internal_size > 0:
            h, w = image.shape[-2:]
            min_side = min(h, w)
            if min_side > self.max_internal_size:
                resize_needed = True
                new_h = int(h / min_side * self.max_internal_size)
                new_w = int(w / min_side * self.max_internal_size)
                image = F.interpolate(image.unsqueeze(0),
                                      size=(new_h, new_w),
                                      mode='bilinear',
                                      align_corners=False)[0]
                if mask is not None:
                    if idx_mask:
                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                             size=(new_h, new_w),
                                             mode='nearest',
                                             align_corners=False)[0, 0].round().long()
                    else:
                        mask = F.interpolate(mask.unsqueeze(0),
                                             size=(new_h, new_w),
                                             mode='bilinear',
                                             align_corners=False)[0]

        self.curr_ti += 1

        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        if self.flip_aug:
            image = torch.cat([image, torch.flip(image, dims=[-1])], dim=0)

        # whether to update the working memory
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # segment when there is no input mask or when the input mask is incomplete
        need_segment = (mask is None) or (self.object_manager.num_obj > 0
                                          and not self.object_manager.has_all(objects))
        update_sensory = ((self.curr_ti - self.last_mem_ti) in self.stagger_ti) and (not end)

        # encoding the image
        ms_feat, pix_feat = self.image_feature_store.get_features(self.curr_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(self.curr_ti, image)

        # segmentation from memory if needed
        if need_segment:
            pred_prob_with_bg = self._segment(key,
                                              selection,
                                              pix_feat,
                                              ms_feat,
                                              update_sensory=update_sensory)

        # use the input mask if provided
        if mask is not None:
            # inform the manager of the new objects, and get a list of temporary id
            # temporary ids -- indicates the position of objects in the tensor
            # (starts with 1 due to the background channel)
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)

            mask, _ = pad_divide_by(mask, 16)
            if need_segment:
                # merge predicted mask with the incomplete input mask
                pred_prob_no_bg = pred_prob_with_bg[1:]
                # use the mutual exclusivity of segmentation
                if idx_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.max(0) > 0.5] = 0

                new_masks = []
                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                    if idx_mask:
                        this_mask = (mask == objects[mask_id]).type_as(pred_prob_no_bg)
                    else:
                        this_mask = mask[tmp_id]
                    if tmp_id >= pred_prob_no_bg.shape[0]:
                        new_masks.append(this_mask.unsqueeze(0))
                    else:
                        # +1 for padding the background channel
                        pred_prob_no_bg[tmp_id + 1] = this_mask
                # new_masks are always in the order of tmp_id
                mask = torch.cat([pred_prob_no_bg, *new_masks], dim=0)
            elif idx_mask:
                # simply convert cls to one-hot representation
                if len(objects) == 0:
                    if delete_buffer:
                        self.image_feature_store.delete(self.curr_ti)
                    log.warn('Trying to insert an empty mask as memory!')
                    return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                                       device=key.device,
                                       dtype=key.dtype)
                mask = torch.stack(
                    [mask == objects[mask_id] for mask_id, _ in enumerate(corresponding_tmp_ids)],
                    dim=0)
            pred_prob_with_bg = aggregate(mask, dim=0)
            pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0)

        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)
        if self.flip_aug:
            self.last_mask = torch.cat(
                [self.last_mask, torch.flip(self.last_mask, dims=[-1])], dim=0)

        # save as memory if needed
        if is_mem_frame or force_permanent:
            self._add_memory(image,
                             pix_feat,
                             self.last_mask,
                             key,
                             shrinkage,
                             selection,
                             force_permanent=force_permanent)

        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti)

        output_prob = unpad(pred_prob_with_bg, self.pad)
        if resize_needed:
            # restore output to the original size
            output_prob = F.interpolate(output_prob.unsqueeze(0),
                                        size=(h, w),
                                        mode='bilinear',
                                        align_corners=False)[0]

        return output_prob

    def get_aux_outputs(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image, pads = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        _, pix_feat = self.image_feature_store.get_features(self.curr_ti, image)

        aux_inputs = self.memory.aux
        aux_outputs = self.network.compute_aux(pix_feat, aux_inputs, selector=None)
        aux_outputs['q_weights'] = aux_inputs['q_weights']
        aux_outputs['p_weights'] = aux_inputs['p_weights']

        for k, v in aux_outputs.items():
            if len(v.shape) == 5:
                aux_outputs[k] = F.interpolate(v[0],
                                               size=image.shape[-2:],
                                               mode='bilinear',
                                               align_corners=False)
            elif 'weights' in k:
                b, num_objects, num_heads, num_queries, h, w = v.shape
                v = v.view(num_objects * num_heads, num_queries, h, w)
                v = F.interpolate(v, size=image.shape[-2:], mode='bilinear', align_corners=False)
                aux_outputs[k] = v.view(num_objects, num_heads, num_queries, *image.shape[-2:])
            else:
                aux_outputs[k] = F.interpolate(v,
                                               size=image.shape[-2:],
                                               mode='bilinear',
                                               align_corners=False)[0]
            aux_outputs[k] = unpad(aux_outputs[k], pads)
            if 'weights' in k:
                weights = aux_outputs[k]
                weights = weights / (weights.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] +
                                     1e-8)
                aux_outputs[k] = (weights * 255).cpu().numpy()
            else:
                aux_outputs[k] = (aux_outputs[k].softmax(dim=0) * 255).cpu().numpy()

        self.image_feature_store.delete(self.curr_ti)
        return aux_outputs

    def get_aux_object_weights(self, image: torch.Tensor) -> np.ndarray:
        image, pads = pad_divide_by(image, 16)
        # B*num_objects*H*W*num_queries -> num_objects*num_queries*H*W
        # weights = F.softmax(self.obj_logits, dim=-1)[0]
        weights = F.sigmoid(self.obj_logits)[0]
        weights = weights.permute(0, 3, 1, 2).contiguous()
        weights = F.interpolate(weights,
                                size=image.shape[-2:],
                                mode='bilinear',
                                align_corners=False)
        # weights = weights / (weights.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0])
        weights = unpad(weights, pads)
        weights = (weights * 255).cpu().numpy()
        return weights
