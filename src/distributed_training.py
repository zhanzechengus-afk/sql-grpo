from contextlib import contextmanager
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank():
    return dist.get_rank() if world_size() > 1 else 0


@contextmanager
def distributed_training(requested_device, *, predict=False):
    count = int(os.environ.get("WORLD_SIZE", "1"))
    if count == 1:
        yield requested_device
        return
    if predict:
        raise ValueError("Use a single process for prediction")
    if count < 1 or torch.device(requested_device).type != "cuda":
        raise ValueError("Distributed training requires one CUDA device per process")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    if torch.device(requested_device).index not in (None, local_rank):
        raise ValueError("Use --device cuda and let torchrun select each worker's device")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    try:
        yield str(device)
    finally:
        dist.destroy_process_group()


def shard(rows):
    return rows[rank()::world_size()]


def barrier():
    if world_size() > 1:
        dist.barrier()


def sum_tensor_(tensor):
    if world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def max_tensor_(tensor):
    if world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


@torch.no_grad()
def reduce_gradients(parameters, weight=1.0):
    for parameter in parameters:
        if parameter.grad is None:
            contribution = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            contribution = parameter.grad.detach().float().mul(weight)
        sum_tensor_(contribution)
        if parameter.grad is None:
            parameter.grad = contribution.to(parameter.dtype)
        else:
            parameter.grad.copy_(contribution)
