import numpy as np
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import ProInpainter


class TrackingAnything():
    def __init__(self, sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)
        self.cutie = BaseTracker(cutie_checkpoint, device=args.device)
        self.baseinpainter = ProInpainter(propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args.device) 
       
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    def generator(self, images: list, template_mask:np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i==0:           
                mask, logit, painted_image = self.cutie.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
            else:
                mask, logit, painted_image = self.cutie.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
        
        
    
    
    