# -*- coding: utf-8 -*-
import sys 
sys.path.append(".") 

import cv2
import os
import numpy as np
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader

from core.dataset import TestDataset
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

from RAFT.utils.flow_viz_pt import flow_to_image

import cvbase
import imageio
from time import time

import warnings
warnings.filterwarnings("ignore")

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_flows(output, videoFlowF, videoFlowB):
    # create_dir(os.path.join(output, 'forward_flo'))
    # create_dir(os.path.join(output, 'backward_flo'))
    create_dir(os.path.join(output, 'forward_png'))
    create_dir(os.path.join(output, 'backward_png'))
    N = videoFlowF.shape[-1]
    for i in range(N):
        forward_flow = videoFlowF[..., i]
        backward_flow = videoFlowB[..., i]
        forward_flow_vis = cvbase.flow2rgb(forward_flow)
        backward_flow_vis = cvbase.flow2rgb(backward_flow)
        # cvbase.write_flow(forward_flow, os.path.join(output,  'forward_flo', '{:05d}.flo'.format(i)))
        # cvbase.write_flow(backward_flow, os.path.join(output,  'backward_flo', '{:05d}.flo'.format(i)))
        forward_flow_vis = (forward_flow_vis*255.0).astype(np.uint8)
        backward_flow_vis = (backward_flow_vis*255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(output,  'forward_png', '{:05d}.png'.format(i)), forward_flow_vis)
        imageio.imwrite(os.path.join(output,  'backward_png', '{:05d}.png'.format(i)), backward_flow_vis)

def tensor2np(array):
    array = torch.stack(array, dim=-1).squeeze(0).permute(1, 2, 0, 3).cpu().numpy()
    return array

def main_worker(args):
    # set up datasets and data loader
    args.size = (args.width, args.height)
    test_dataset = TestDataset(vars(args))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_raft = RAFT_bi(args.raft_model_path, device)
    
    fix_flow_complete = RecurrentFlowCompleteNet(args.fc_model_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    total_frame_epe = []
    time_all = []

    print('Start evaluation...')
    # create results directory
    result_path = os.path.join('results_flow', f'{args.dataset}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    eval_summary = open(os.path.join(result_path, f"{args.dataset}_metrics.txt"), "w")

    for index, items in enumerate(test_loader):
        frames, masks, flows_f, flows_b, video_name, frames_PIL = items
        local_masks = masks.float().to(device)

        video_length = frames.size(1)
        
        if args.load_flow:
            gt_flows_bi = (flows_f.to(device), flows_b.to(device))
        else:
            short_len = 60
            if frames.size(1) > short_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_len):
                    end_f = min(video_length, f + short_len)
                    if f == 0:
                        flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=args.raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = fix_raft(frames, iters=20)

        torch.cuda.synchronize()
        time_start = time()

        # flow_length = flows_f.size(1)
        # f_stride = 30
        # pred_flows_f = []
        # pred_flows_b = []
        # suffix = flow_length%f_stride
        # last = flow_length//f_stride
        # for f in range(0, flow_length, f_stride):
        #     gt_flows_bi_i = (gt_flows_bi[0][:,f:f+f_stride], gt_flows_bi[1][:,f:f+f_stride])
        #     pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi_i, local_masks[:,f:f+f_stride+1])
        #     pred_flows_f_i, pred_flows_b_i = fix_flow_complete.combine_flow(gt_flows_bi_i, pred_flows_bi, local_masks[:,f:f+f_stride+1])
        #     pred_flows_f.append(pred_flows_f_i)
        #     pred_flows_b.append(pred_flows_b_i)
        # pred_flows_f = torch.cat(pred_flows_f, dim=1)
        # pred_flows_b = torch.cat(pred_flows_b, dim=1)
        # pred_flows_bi = (pred_flows_f, pred_flows_b)

        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, local_masks)
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, local_masks)

        torch.cuda.synchronize()
        time_i = time() - time_start
        time_i = time_i*1.0/frames.size(1)

        time_all = time_all+[time_i]*frames.size(1)

        cur_video_epe = []
        
        epe1 = torch.mean(torch.sum((flows_f - pred_flows_bi[0].cpu())**2, dim=2).sqrt())
        epe2 = torch.mean(torch.sum((flows_b - pred_flows_bi[1].cpu())**2, dim=2).sqrt())

        cur_video_epe.append(epe1.numpy())
        cur_video_epe.append(epe2.numpy())

        total_frame_epe = total_frame_epe+[epe1.numpy()]*flows_f.size(1)
        total_frame_epe = total_frame_epe+[epe2.numpy()]*flows_f.size(1)

        cur_epe = sum(cur_video_epe) / len(cur_video_epe)
        avg_time = sum(time_all) / len(time_all)
        print(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | EPE: {cur_epe:.4f} | Time: {avg_time:.4f}'
        )
        eval_summary.write(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | EPE: {cur_epe:.4f} | Time: {avg_time:.4f}\n'
        )

        # saving images for evaluating warpping errors
        if args.save_results:
            forward_flows = pred_flows_bi[0].cpu().permute(1,0,2,3,4)
            backward_flows = pred_flows_bi[1].cpu().permute(1,0,2,3,4)
            # forward_flows = flows_f.cpu().permute(1,0,2,3,4)
            # backward_flows = flows_b.cpu().permute(1,0,2,3,4)
            videoFlowF = list(forward_flows)
            videoFlowB = list(backward_flows)

            videoFlowF = tensor2np(videoFlowF)
            videoFlowB = tensor2np(videoFlowB)

            save_frame_path = os.path.join(result_path, video_name[0])
            save_flows(save_frame_path, videoFlowF, videoFlowB)

    avg_frame_epe = sum(total_frame_epe) / len(total_frame_epe)

    print(f'Finish evaluation... Average Frame EPE: {avg_frame_epe:.4f} | | Time: {avg_time:.4f}')
    eval_summary.write(f'Finish evaluation... Average Frame EPE: {avg_frame_epe:.4f} | | Time: {avg_time:.4f}\n')
    eval_summary.close()
    
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
