"""
InsightSerenity AI Engine — Distributed Training Wrapper
=========================================================
Wraps a model in PyTorch's DistributedDataParallel (DDP) for multi-GPU
training. DDP replicates the model on each GPU and averages gradients
across all replicas after each backward pass.

Why DDP over DataParallel?
    - DataParallel: single process, single machine, GIL bottleneck
    - DDP: one process per GPU, no GIL, scales across machines (multi-node)
    DDP is always faster when you have > 1 GPU.

This module provides:
    1. setup_distributed()  — initialise the process group
    2. wrap_model_ddp()     — wrap model and move to correct GPU
    3. cleanup_distributed() — tear down the process group
    4. is_main_process()    — True only on rank 0 (for logging/saving)
    5. DistributedSampler   — re-export for convenience

DDP launch (from terminal):
    torchrun --nproc_per_node=4 scripts/training/train.py --distributed

Or manually:
    python -m torch.distributed.launch --nproc_per_node=4 train.py
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """
    Initialise the distributed process group.

    Must be called at the start of each process before any CUDA operations.
    Reads RANK, LOCAL_RANK, and WORLD_SIZE from environment variables
    (set automatically by torchrun).

    Args:
        backend:     Communication backend. "nccl" for GPU, "gloo" for CPU.
        init_method: How processes find each other. "env://" reads from
                     MASTER_ADDR and MASTER_PORT environment variables.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available on this system")

    rank       = int(os.environ.get("RANK",       "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    # Set the current process to use its dedicated GPU
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    logger.info(
        "Distributed process group initialised",
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group. Call at the end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def wrap_model_ddp(
    model: nn.Module,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> DDP:
    """
    Wrap a model in DistributedDataParallel.

    Must be called AFTER setup_distributed() and AFTER moving the model
    to the correct GPU with model.to(device).

    Args:
        model:                   The model to wrap.
        find_unused_parameters:  Set True if some parameters have no gradient
                                 (e.g. frozen layers or MoE routing bypasses).
        gradient_as_bucket_view: Memory optimisation — gradients are stored
                                 as views into the DDP communication buckets.

    Returns:
        DDP-wrapped model with the same forward interface.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )

    logger.info(
        "Model wrapped in DDP",
        local_rank=local_rank,
        world_size=get_world_size(),
    )
    return wrapped


def is_main_process() -> bool:
    """
    Return True if this process is rank 0 (the main / leader process).

    Only the main process should:
        - Log to console and files
        - Save checkpoints
        - Run evaluation reporting
    All other processes should skip these operations.
    """
    if not dist.is_initialized():
        return True   # Single-process training → always main
    return dist.get_rank() == 0


def get_rank() -> int:
    """Return the global rank of this process (0-indexed)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the total number of processes in the training job."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """
    Block until all processes have reached this point.
    Used to synchronise after a checkpoint write before continuing.
    """
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across all ranks.

    Used to aggregate per-GPU metrics (e.g. loss) into a single global mean.
    Each rank's tensor is summed across all ranks, then divided by world_size.

    Args:
        tensor: Scalar or 1D tensor to average.

    Returns:
        Averaged tensor with the same shape.
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / get_world_size()


# Re-export for convenience — callers import everything from this module
__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_ddp",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "barrier",
    "all_reduce_mean",
    "DistributedSampler",
]
