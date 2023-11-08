"""
Integrate numerical values for some iterations
Typically used for loss computation / logging to tensorboard
Call finalize and create a new Integrator when you want to display/log
"""
from typing import Dict, Callable, Tuple
import torch
from tracker.utils.logger import TensorboardLogger


class Integrator:
    def __init__(self, logger: TensorboardLogger, distributed: bool = True):
        self.values = {}
        self.counts = {}
        self.hooks = []  # List is used here to maintain insertion order

        self.logger = logger

        self.distributed = distributed
        self.local_rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_tensor(self, key: str, tensor: torch.Tensor):
        if key not in self.values:
            self.counts[key] = 1
            if type(tensor) == float or type(tensor) == int:
                self.values[key] = tensor
            else:
                self.values[key] = tensor.mean().item()
        else:
            self.counts[key] += 1
            if type(tensor) == float or type(tensor) == int:
                self.values[key] += tensor
            else:
                self.values[key] += tensor.mean().item()

    def add_dict(self, tensor_dict: Dict[str, torch.Tensor]):
        for k, v in tensor_dict.items():
            self.add_tensor(k, v)

    def add_hook(self, hook: Callable[[torch.Tensor], Tuple[str, torch.Tensor]]):
        """
        Adds a custom hook, i.e. compute new metrics using values in the dict
        The hook takes the dict as argument, and returns a (k, v) tuple
        e.g. for computing IoU
        """
        if type(hook) == list:
            self.hooks.extend(hook)
        else:
            self.hooks.append(hook)

    def reset_except_hooks(self):
        self.values = {}
        self.counts = {}

    # Average and output the metrics
    def finalize(self, exp_id: str, prefix: str, it: int) -> None:

        for hook in self.hooks:
            k, v = hook(self.values)
            self.add_tensor(k, v)

        outputs = {}
        for k, v in self.values.items():

            if k[:4] == 'hide':
                continue

            avg = v / self.counts[k]

            if self.distributed:
                # Inplace operation
                avg = torch.tensor(avg).cuda()
                torch.distributed.reduce(avg, dst=0)

                if self.local_rank == 0:
                    avg = (avg / self.world_size).cpu().item()
                    outputs[k] = avg
            else:
                # Simple does it
                outputs[k] = avg

        if (not self.distributed) or (self.local_rank == 0):
            self.logger.log_metrics(exp_id, prefix, outputs, it)
