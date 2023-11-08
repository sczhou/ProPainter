import cv2
import numpy as np

import torch
from collections import defaultdict


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np


def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np


def detach_to_cpu(x):
    return x.detach().cpu()


def transpose_np(x):
    return np.transpose(x, [1, 2, 0])


def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x


def tensor_to_im(x):
    x = detach_to_cpu(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x


# Predefined key <-> caption dict
key_captions = {
    'im': 'Image',
    'gt': 'GT',
}
"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""


def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w * cate_counts, h * (rows_counts + 1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt * w + 100 + i * dy), font, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt + 0) * w:(col_cnt + 1) * w,
                         (row_cnt + 1) * h:(row_cnt + 2) * h, :] = img

        col_cnt += 1

    return output_image


def base_transform(im, size):
    im = tensor_to_np_float(im)
    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    # Resize
    if im.shape[1] != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

    return im.clip(0, 1)


def im_transform(im, size):
    return base_transform(detach_to_cpu(im), size=size)


def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)


def logits_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)


def add_attention(mask, pos):
    mask = mask[:, :, None].repeat(3, axis=2)
    pos = (pos + 1) / 2
    for i in range(pos.shape[0]):
        y = int(pos[i][0] * mask.shape[0])
        x = int(pos[i][1] * mask.shape[1])
        y = max(min(y, mask.shape[0] - 1), 0)
        x = max(min(x, mask.shape[1] - 1), 0)
        # mask[y, x, :] = (255, 0, 0)
        cv2.circle(mask, (x, y), 5, (1, 0, 0), -1)
    return mask


def vis(images, size, num_objects):
    req_images = defaultdict(list)

    b, t = images['rgb'].shape[:2]

    # limit the number of images saved
    b = min(2, b)

    # find max num objects
    max_num_objects = max(num_objects[:b])

    GT_suffix = ''
    for bi in range(b):
        GT_suffix += ' \n%s' % images['info']['name'][bi][-25:-4]

    for bi in range(b):
        for ti in range(t):
            req_images['RGB'].append(im_transform(images['rgb'][bi, ti], size))
            aux = images[f'aux_{max(ti, 1)}']  # no aux_0, use aux_1 for shape
            if 'sensory_logits' in aux:
                sensory_aux = aux['sensory_logits'][bi].softmax(dim=0)
            # batch_size * num_objects * num_levels * H * W
            q_mask_aux = aux['q_logits'][bi].softmax(dim=0)
            num_levels = q_mask_aux.shape[1]

            for oi in range(max_num_objects):
                if ti == 0 or oi >= num_objects[bi]:
                    req_images[f'Mask_{oi}'].append(
                        mask_transform(images['first_frame_gt'][bi][0, oi], size))
                    req_images[f'S-Aux_{oi}'].append(
                        mask_transform(images['first_frame_gt'][bi][0, oi], size))
                    for l in range(num_levels):
                        req_images[f'Q-Aux-L{l}_{oi}'].append(
                            mask_transform(images['first_frame_gt'][bi][0, oi], size))
                else:
                    mask = mask_transform(images[f'masks_{ti}'][bi][oi], size)
                    req_images[f'Mask_{oi}'].append(mask)
                    if 'sensory_logits' in aux:
                        req_images[f'S-Aux_{oi}'].append(mask_transform(sensory_aux[oi + 1], size))

                    for l in range(num_levels):
                        mask = mask_transform(q_mask_aux[oi + 1, l], size)
                        req_images[f'Q-Aux-L{l}_{oi}'].append(mask)

                req_images[f'GT_{oi}_{GT_suffix}'].append(
                    mask_transform(images['cls_gt'][bi, ti, 0] == (oi + 1), size))

    return get_image_array(req_images, size, key_captions)


def vis_debug(images, size, num_objects):
    req_images = defaultdict(list)

    b, t = images['rgb'].shape[:2]

    # limit the number of images saved
    b = min(2, b)

    # find max num objects
    max_num_objects = max(num_objects[:b])

    GT_suffix = ''
    for bi in range(b):
        GT_suffix += ' \n%s' % images['info']['name'][bi][-25:-4]

    for bi in range(b):
        for ti in range(t):
            req_images['RGB'].append(im_transform(images['rgb'][bi, ti], size))
            aux = images[f'aux_{max(ti, 1)}']  # no aux_0, use aux_1 for shape
            sensory_aux = aux['sensory_logits'][bi].softmax(dim=0)
            # batch_size * num_objects * num_levels * H * W
            q_mask_aux = aux['q_logits'][bi].softmax(dim=0)
            attn_mask = aux['attn_mask'][bi]
            num_levels = q_mask_aux.shape[1]
            num_queries = attn_mask.shape[1]

            for oi in range(max_num_objects):
                if ti == 0 or oi >= num_objects[bi]:
                    req_images[f'Mask_{oi}'].append(
                        mask_transform(images['first_frame_gt'][bi][0, oi], size))
                    req_images[f'S-Aux_{oi}'].append(
                        mask_transform(images['first_frame_gt'][bi][0, oi], size))
                    for l in range(num_levels):
                        req_images[f'Q-Aux-L{l}_{oi}'].append(
                            mask_transform(images['first_frame_gt'][bi][0, oi], size))
                    for q in range(num_queries):
                        req_images[f'Attn-Mask-Q{q}_{oi}'].append(
                            mask_transform(images['first_frame_gt'][bi][0, oi], size))
                else:
                    mask = mask_transform(images[f'masks_{ti}'][bi][oi], size)
                    req_images[f'Mask_{oi}'].append(mask)
                    req_images[f'S-Aux_{oi}'].append(mask_transform(sensory_aux[oi + 1], size))

                    for l in range(num_levels):
                        mask = mask_transform(q_mask_aux[oi + 1, l], size)
                        req_images[f'Q-Aux-L{l}_{oi}'].append(mask)
                    for q in range(num_queries):
                        mask = mask_transform(1 - attn_mask[oi, q].float(), size)
                        req_images[f'Attn-Mask-Q{q}_{oi}'].append(mask)

                req_images[f'GT_{oi}_{GT_suffix}'].append(
                    mask_transform(images['cls_gt'][bi, ti, 0] == (oi + 1), size))

    return get_image_array(req_images, size, key_captions)