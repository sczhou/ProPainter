# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import warnings

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import imageio

# Local imports
from core.dataset import TestDataset
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from RAFT.utils.flow_viz_pt import flow_to_image
import cvbase

warnings.filterwarnings("ignore")

# ... (rest of the code remains the same)

def save_flows(output, videoFlowF, videoFlowB):
    create_dir(os.path.join(output, 'forward_png'))
    create_dir(os.path.join(output, 'backward_png'))
    num_frames = videoFlowF.shape[-1]
    for i in range(num_frames):
        forward_flow = videoFlowF[..., i]
        backward_flow = videoFlowB[..., i]
        forward_flow_vis = cvbase.flow2rgb(forward_flow)
        backward_flow_vis = cvbase.flow2rgb(backward_flow)
        forward_flow_vis = (forward_flow_vis * 255.0).astype(np.uint8)
        backward_flow_vis = (backward_flow_vis * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(output, 'forward_png', f'{i:05d}.png'), forward_flow_vis)
        imageio.imwrite(os.path.join(output, 'backward_png', f'{i:05d}.png'), backward_flow_vis)

# ... (rest of the code remains the same)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument('--raft_model_path', default='weights/raft-things.pth', type=str)
    parser.add_argument('--fc_model_path', default='weights/recurrent_flow_completion.pth', type=str)
    parser.add_argument('--dataset', choices=['davis', 'youtube-vos'], type=str)
    parser.add_argument('--video_root', default='dataset_root', type=str)
    parser.add_argument('--mask_root', default='mask_root', type=str)
    parser.add_argument('--flow_root', default='flow_ground_truth_root', type=str)
    parser.add_argument('--load_flow', default=False, type=bool)
    parser.add_argument("--raft_iter", type=int, default=20)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()
    main_worker(args)

