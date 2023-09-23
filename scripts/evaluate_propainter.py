# -*- coding: utf-8 -*-
import sys 
sys.path.append(".") 

import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator

# from core.dataset import TestDataset
from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

from time import time

import warnings
warnings.filterwarnings("ignore")

# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, ref_stride=10):
    ref_index = []
    for i in range(0, length, ref_stride):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


def main_worker(args):
    args.size = (args.width, args.height)
    w, h = args.size    
    # set up datasets and data loader
    assert (args.dataset == 'davis') or args.dataset == 'youtube-vos', \
        f"{args.dataset} dataset is not supported"
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

    model = InpaintGenerator(model_path=args.propainter_model_path).to(device)
    model.eval()

    time_all = []


    print('Start evaluation ...')
    if args.task == 'video_completion':
        result_path = os.path.join(f'results_eval', 
            f'{args.dataset}_rs_{args.ref_stride}_nl_{args.neighbor_length}_video_completion')
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        eval_summary = open(os.path.join(result_path, f"{args.dataset}_metrics.txt"),"w")
        total_frame_psnr = []
        total_frame_ssim = []
        output_i3d_activations = []
        real_i3d_activations = []
        i3d_model = init_i3d_model('weights/i3d_rgb_imagenet.pt')
    else:
        result_path = os.path.join(f'results_eval', 
            f'{args.dataset}_rs_{args.ref_stride}_nl_{args.neighbor_length}_object_removal')
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)        

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
        
    for index, items in enumerate(test_loader):
        torch.cuda.empty_cache()

        # frames, masks, video_name, frames_PIL = items
        frames, masks, flows_f, flows_b, video_name, frames_PIL = items
        video_name = video_name[0]
        print('Processing:', video_name)

        video_length = frames.size(1)
        frames, masks = frames.to(device), masks.to(device)
        masked_frames = frames * (1 - masks)

        torch.cuda.synchronize()
        time_start = time()

        with torch.no_grad():
            # ---- compute flow ----
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
                    gt_flows_bi = fix_raft(frames, iters=args.raft_iter)

            # ---- complete flow ----
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, masks)
        
            # ---- temporal propagation ----
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks, 'nearest')

            b, t, _, _, _ = masks.size()
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            updated_frames = frames * (1-masks) + prop_imgs.view(b, t, 3, h, w) * masks # merge
            
            del gt_flows_bi, frames, updated_local_masks
            if not args.load_flow:
                torch.cuda.empty_cache()

        ori_frames = frames_PIL
        ori_frames = [
            ori_frames[i].squeeze().cpu().numpy() for i in range(video_length)
        ]
        comp_frames = [None] * video_length

        # complete holes by our model
        neighbor_stride = args.neighbor_length // 2
        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(neighbor_ids, video_length, args.ref_stride)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)


                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    

        torch.cuda.synchronize()
        time_i = time() - time_start
        time_i = time_i*1.0/video_length
        time_all.append(time_i)

        if args.task == 'video_completion':
            # calculate metrics
            cur_video_psnr = []
            cur_video_ssim = []
            comp_PIL = []  # to calculate VFID
            frames_PIL = []
            for ori, comp in zip(ori_frames, comp_frames):
                psnr, ssim = calc_psnr_and_ssim(ori, comp)

                cur_video_psnr.append(psnr)
                cur_video_ssim.append(ssim)

                total_frame_psnr.append(psnr)
                total_frame_ssim.append(ssim)

                frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
                comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))

            # saving i3d activations
            frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL,
                                                            comp_PIL,
                                                            i3d_model,
                                                            device=device)
            real_i3d_activations.append(frames_i3d)
            output_i3d_activations.append(comp_i3d)

            cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
            cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)

            avg_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
            avg_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

            avg_time = sum(time_all) / len(time_all)
            print(
                f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f} \
                    | Avg PSNR/SSIM: {avg_psnr:.4f}/{avg_ssim:.4f} | Time: {avg_time:.4f}'
            )
            eval_summary.write(
                f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f} \
                    | Avg PSNR/SSIM: {avg_psnr:.4f}/{avg_ssim:.4f} | Time: {avg_time:.4f}\n'
            )
        else:
            avg_time = sum(time_all) / len(time_all)
            print(
                f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | Time: {avg_time:.4f}'
            )

        # saving images for evaluating warpping errors
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name)
            if not os.path.exists(save_frame_path):
                os.makedirs(save_frame_path, exist_ok=False)

            for i, frame in enumerate(comp_frames):
                cv2.imwrite(
                    os.path.join(save_frame_path,
                                 str(i).zfill(5) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
    if args.task == 'video_completion':
        avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
        avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

        fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
        print('Finish evaluation... Average Frame PSNR/SSIM/VFID: '
            f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f} | Time: {avg_time:.4f}')
        eval_summary.write(
            'Finish evaluation... Average Frame PSNR/SSIM/VFID: '
            f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f} | Time: {avg_time:.4f}')
        eval_summary.close()
    else:
        print('Finish evaluation... Time: {avg_time:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=20)
    parser.add_argument("--raft_iter", type=int, default=20)
    parser.add_argument('--task', default='video_completion', choices=['object_removal', 'video_completion'])
    parser.add_argument('--raft_model_path', default='weights/raft-things.pth', type=str)
    parser.add_argument('--fc_model_path', default='weights/recurrent_flow_completion.pth', type=str)
    parser.add_argument('--propainter_model_path', default='weights/ProPainter.pth', type=str)
    parser.add_argument('--dataset', choices=['davis', 'youtube-vos'], type=str)
    parser.add_argument('--video_root', default='dataset_root', type=str)
    parser.add_argument('--mask_root', default='mask_root', type=str)
    parser.add_argument('--flow_root', default='flow_ground_truth_root', type=str)
    parser.add_argument('--load_flow', default=False, type=bool)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()
    main_worker(args)
