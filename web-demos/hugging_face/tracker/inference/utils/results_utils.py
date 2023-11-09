from typing import Tuple, Optional, Dict
import logging
import os
import shutil
from os import path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

import pycocotools.mask as mask_util
from threading import Thread
from queue import Queue
from dataclasses import dataclass
import copy

from tracker.utils.pano_utils import ID2RGBConverter
from tracker.utils.palette import davis_palette_np
from tracker.inference.object_manager import ObjectManager
from tracker.inference.object_info import ObjectInfo

log = logging.getLogger()

try:
    import hickle as hkl
except ImportError:
    log.warning('Failed to import hickle. Fine if not using multi-scale testing.')


class ResultSaver:
    def __init__(self,
                 output_root,
                 video_name,
                 *,
                 dataset,
                 object_manager: ObjectManager,
                 use_long_id,
                 palette=None,
                 save_mask=True,
                 save_scores=False,
                 score_output_root=None,
                 visualize_output_root=None,
                 visualize=False,
                 init_json=None):
        self.output_root = output_root
        self.video_name = video_name
        self.dataset = dataset.lower()
        self.use_long_id = use_long_id
        self.palette = palette
        self.object_manager = object_manager
        self.save_mask = save_mask
        self.save_scores = save_scores
        self.score_output_root = score_output_root
        self.visualize_output_root = visualize_output_root
        self.visualize = visualize

        if self.visualize:
            if self.palette is not None:
                self.colors = np.array(self.palette, dtype=np.uint8).reshape(-1, 3)
            else:
                self.colors = davis_palette_np

        self.need_remapping = True
        self.json_style = None
        self.id2rgb_converter = ID2RGBConverter()

        if 'burst' in self.dataset:
            assert init_json is not None
            self.input_segmentations = init_json['segmentations']
            self.segmentations = [{} for _ in init_json['segmentations']]
            self.annotated_frames = init_json['annotated_image_paths']
            self.video_json = {k: v for k, v in init_json.items() if k != 'segmentations'}
            self.video_json['segmentations'] = self.segmentations
            self.json_style = 'burst'

        self.queue = Queue(maxsize=10)
        self.thread = Thread(target=save_result, args=(self.queue, ))
        self.thread.daemon = True
        self.thread.start()

    def process(self,
                prob: torch.Tensor,
                frame_name: str,
                resize_needed: bool = False,
                shape: Optional[Tuple[int, int]] = None,
                last_frame: bool = False,
                path_to_image: str = None):

        if resize_needed:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
                                                                                                 0]
        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)
        if self.save_scores:
            # also need to pass prob
            prob = prob.cpu()
        else:
            prob = None

        # remap indices
        if self.need_remapping:
            new_mask = torch.zeros_like(mask)
            for tmp_id, obj in self.object_manager.tmp_id_to_obj.items():
                new_mask[mask == tmp_id] = obj.id
            mask = new_mask

        args = ResultArgs(saver=self,
                          prob=prob,
                          mask=mask.cpu(),
                          frame_name=frame_name,
                          path_to_image=path_to_image,
                          tmp_id_to_obj=copy.deepcopy(self.object_manager.tmp_id_to_obj),
                          obj_to_tmp_id=copy.deepcopy(self.object_manager.obj_to_tmp_id),
                          last_frame=last_frame)

        self.queue.put(args)

    def end(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


@dataclass
class ResultArgs:
    saver: ResultSaver
    prob: torch.Tensor
    mask: torch.Tensor
    frame_name: str
    path_to_image: str
    tmp_id_to_obj: Dict[int, ObjectInfo]
    obj_to_tmp_id: Dict[ObjectInfo, int]
    last_frame: bool


def save_result(queue: Queue):
    while True:
        args: ResultArgs = queue.get()
        if args is None:
            queue.task_done()
            break

        saver = args.saver
        prob = args.prob
        mask = args.mask
        frame_name = args.frame_name
        path_to_image = args.path_to_image
        tmp_id_to_obj = args.tmp_id_to_obj
        obj_to_tmp_id = args.obj_to_tmp_id
        last_frame = args.last_frame
        all_obj_ids = [k.id for k in obj_to_tmp_id]

        # record output in the json file
        if saver.json_style == 'burst':
            if frame_name in saver.annotated_frames:
                frame_index = saver.annotated_frames.index(frame_name)
                input_segments = saver.input_segmentations[frame_index]
                frame_segments = saver.segmentations[frame_index]

                for id in all_obj_ids:
                    if id in input_segments:
                        # if this frame has been given as input, just copy
                        frame_segments[id] = input_segments[id]
                        continue

                    segment = {}
                    segment_mask = (mask == id)
                    if segment_mask.sum() > 0:
                        coco_mask = mask_util.encode(np.asfortranarray(segment_mask.numpy()))
                        segment['rle'] = coco_mask['counts'].decode('utf-8')
                        frame_segments[id] = segment

        # save the mask to disk
        if saver.save_mask:
            if saver.use_long_id:
                out_mask = mask.numpy().astype(np.uint32)
                rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
                for id in all_obj_ids:
                    _, image = saver.id2rgb_converter.convert(id)
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = image
                out_img = Image.fromarray(rgb_mask)
            else:
                rgb_mask = None
                out_mask = mask.numpy().astype(np.uint8)
                out_img = Image.fromarray(out_mask)
                if saver.palette is not None:
                    out_img.putpalette(saver.palette)

            this_out_path = path.join(saver.output_root, saver.video_name)
            os.makedirs(this_out_path, exist_ok=True)
            out_img.save(os.path.join(this_out_path, frame_name[:-4] + '.png'))

        # save scores for multi-scale testing
        if saver.save_scores:
            this_out_path = path.join(saver.score_output_root, saver.video_name)
            os.makedirs(this_out_path, exist_ok=True)

            prob = (prob.detach().numpy() * 255).astype(np.uint8)

            if last_frame:
                tmp_to_obj_mapping = {obj.id: tmp_id for obj, tmp_id in tmp_id_to_obj.items()}
                hkl.dump(tmp_to_obj_mapping, path.join(this_out_path, f'backward.hkl'), mode='w')

            hkl.dump(prob,
                     path.join(this_out_path, f'{frame_name[:-4]}.hkl'),
                     mode='w',
                     compression='lzf')

        if saver.visualize:
            if path_to_image is not None:
                image_np = np.array(Image.open(path_to_image))
            else:
                raise ValueError('Cannot visualize without path_to_image')

            if rgb_mask is None:
                # we need to apply a palette
                rgb_mask = np.zeros((*out_mask.shape, 3), dtype=np.uint8)
                for id in all_obj_ids:
                    image = saver.colors[id]
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = image

            alpha = (out_mask == 0).astype(np.float32) * 0.5 + 0.5
            alpha = alpha[:, :, None]
            blend = (image_np * alpha + rgb_mask * (1 - alpha)).astype(np.uint8)

            # find a place to save the visualization
            this_vis_path = path.join(saver.visualize_output_root, saver.video_name)
            os.makedirs(this_vis_path, exist_ok=True)
            Image.fromarray(blend).save(path.join(this_vis_path, frame_name[:-4] + '.jpg'))

        queue.task_done()


def make_zip(dataset, run_dir, exp_id, mask_output_root):
    if dataset.startswith('y'):
        # YoutubeVOS
        log.info('Making zip for YouTubeVOS...')
        shutil.make_archive(path.join(run_dir, f'{exp_id}_{dataset}'), 'zip', run_dir,
                            'Annotations')
    elif dataset == 'd17-test-dev':
        # DAVIS 2017 test-dev -- zip from within the Annotation folder
        log.info('Making zip for DAVIS test-dev...')
        shutil.make_archive(path.join(run_dir, f'{exp_id}_{dataset}'), 'zip', mask_output_root)
    elif dataset == 'mose-val':
        # MOSE validation -- same as DAVIS test-dev
        log.info('Making zip for MOSE validation...')
        shutil.make_archive(path.join(run_dir, f'{exp_id}_{dataset}'), 'zip', mask_output_root)
    elif dataset == 'lvos-test':
        # LVOS test -- same as YouTubeVOS
        log.info('Making zip for LVOS test...')
        shutil.make_archive(path.join(run_dir, f'{exp_id}_{dataset}'), 'zip', run_dir,
                            'Annotations')
    else:
        log.info(f'Not making zip for {dataset}.')
