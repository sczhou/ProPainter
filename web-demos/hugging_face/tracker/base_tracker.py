import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import sys
sys.path.append('../')

from tracker.config import CONFIG
from tracker.model.cutie import CUTIE
from tracker.inference.inference_core import InferenceCore
from tracker.utils.mask_mapper import MaskMapper

from tools.painter import mask_painter


class BaseTracker:
    def __init__(self, cutie_checkpoint, device) -> None:
        """
        device: model device
        cutie_checkpoint: checkpoint of XMem model
        """
        config = OmegaConf.create(CONFIG)

        # initialise XMem
        network = CUTIE(config).to(device).eval()
        model_weights = torch.load(cutie_checkpoint, map_location=device)
        network.load_weights(model_weights)

        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        self.device = device
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def image_to_torch(self, frame: np.ndarray, device: str = 'cuda'):
            # frame: H*W*3 numpy array
            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
            return frame
    
    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        logit: numpy array (H, W), logit

        Output:
        mask: numpy arrays (H, W)
        logit: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """

        if first_frame_annotation is not None:   # first frame mask
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
        else:
            mask = None
            labels = None

        # prepare inputs
        frame_tensor = self.image_to_torch(frame, self.device)
        
        # track one frame
        probs = self.tracker.step(frame_tensor, mask, labels)   # logits 2 (bg fg) H W

        # convert to mask
        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

        final_mask = np.zeros_like(out_mask)
        
        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs+1):
            if np.max(final_mask==obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask==obj).astype('uint8'), mask_color=obj+1)

        return final_mask, final_mask, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()