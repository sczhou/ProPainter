from typing import List, Dict
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn.functional as F

from tracker.utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from tracker.utils.tensor_utils import cls_to_one_hot


@torch.jit.script
def ce_loss(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    # logits: T*C*num_points
    loss = F.cross_entropy(logits, soft_gt, reduction='none')
    # sum over temporal dimension
    return loss.sum(0).mean()


@torch.jit.script
def dice_loss(mask: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    # mask: T*C*num_points
    # soft_gt: T*C*num_points
    # ignores the background
    mask = mask[:, 1:].flatten(start_dim=2)
    gt = soft_gt[:, 1:].float().flatten(start_dim=2)
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum(0).mean()


class LossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

    def mask_loss(self, logits: torch.Tensor,
                  soft_gt: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.point_supervision

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, lambda x: calculate_uncertainty(x), self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            # get gt labels
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)
        point_logits = point_sample(logits, point_coords, align_corners=False)
        # point_labels and point_logits: B*C*num_points

        loss_ce = ce_loss(point_logits, point_labels)
        loss_dice = dice_loss(point_logits.softmax(dim=1), point_labels)

        return loss_ce, loss_dice

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        t_range = range(1, num_frames)

        for bi in range(batch_size):
            logits = torch.stack([data[f'logits_{ti}'][bi, :num_objects[bi] + 1] for ti in t_range],
                                 dim=0)
            cls_gt = data['cls_gt'][bi, 1:]  # remove gt for the first frame
            soft_gt = cls_to_one_hot(cls_gt, num_objects[bi])

            loss_ce, loss_dice = self.mask_loss(logits, soft_gt)
            losses['loss_ce'] += loss_ce / batch_size
            losses['loss_dice'] += loss_dice / batch_size

            aux = [data[f'aux_{ti}'] for ti in t_range]
            if 'sensory_logits' in aux[0]:
                sensory_log = torch.stack(
                    [a['sensory_logits'][bi, :num_objects[bi] + 1] for a in aux], dim=0)
                loss_ce, loss_dice = self.mask_loss(sensory_log, soft_gt)
                losses['aux_sensory_ce'] += loss_ce / batch_size * self.sensory_weight
                losses['aux_sensory_dice'] += loss_dice / batch_size * self.sensory_weight
            if 'q_logits' in aux[0]:
                num_levels = aux[0]['q_logits'].shape[2]

                for l in range(num_levels):
                    query_log = torch.stack(
                        [a['q_logits'][bi, :num_objects[bi] + 1, l] for a in aux], dim=0)
                    loss_ce, loss_dice = self.mask_loss(query_log, soft_gt)
                    losses[f'aux_query_ce_l{l}'] += loss_ce / batch_size * self.query_weight
                    losses[f'aux_query_dice_l{l}'] += loss_dice / batch_size * self.query_weight

        losses['total_loss'] = sum(losses.values())

        return losses
