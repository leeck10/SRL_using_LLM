# -*- coding: utf-8 -*-
"""GPU setup utilities.

CRITICAL: ``set_gpu_before_import()`` must be called at the very top of each
script — **before** importing ``torch``, ``transformers``, or ``bitsandbytes``.
These libraries may initialize CUDA during import, after which changing
``CUDA_VISIBLE_DEVICES`` has no effect.

Typical usage at the top of a script::

    import os, sys, argparse, yaml          # safe: no CUDA
    sys.path.insert(0, ...)
    from utils.gpu import set_gpu_before_import
    set_gpu_before_import()                  # reads --gpu / --config, sets env var

    # NOW it is safe to import CUDA-dependent libraries
    import torch
    from transformers import ...
"""

import os
import sys
import argparse


def set_gpu_before_import(argv=None):
    """Parse ``--gpu`` and ``--config`` from CLI, then set CUDA_VISIBLE_DEVICES.

    This function uses a **minimal** argument parser that only looks at
    ``--gpu`` and ``--config``.  It does NOT consume the full argv — call
    ``argparse`` again later for the complete argument set.

    Must be called **before** any ``import torch`` or similar statement.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        Resolved physical GPU id (int).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--config", default=None)
    known, _ = parser.parse_known_args(argv)

    gpu_id = known.gpu  # may be None

    # If --gpu was not given, try to read gpu_id from the config file
    if gpu_id is None and known.config is not None:
        try:
            import yaml  # yaml is pure-Python, no CUDA
            with open(known.config, "r") as f:
                cfg = yaml.safe_load(f)
            gpu_id = int(cfg.get("gpu_id", 0))
        except Exception:
            gpu_id = 0

    if gpu_id is None:
        gpu_id = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU] CUDA_VISIBLE_DEVICES={gpu_id}  (physical GPU {gpu_id})")
    return gpu_id


def setup_gpu(gpu_id: int = 0) -> str:
    """Configure CUDA_VISIBLE_DEVICES and return the logical device string.

    If ``set_gpu_before_import()`` was already called, this is a no-op
    that just returns the device string.  It is kept for backward compatibility
    and for use in library code (e.g., ``inference/pipeline.py``) where
    the caller controls import order.

    Args:
        gpu_id: Physical GPU index to use (e.g., 0, 1, 2, ...).

    Returns:
        Logical device string, always ``"cuda:0"`` when a GPU is requested,
        or ``"cpu"`` if gpu_id is negative.
    """
    if gpu_id < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("GPU disabled — using CPU")
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return "cuda:0"


def resolve_gpu_id(args_gpu, cfg_gpu_id: int = 0) -> int:
    """Determine which GPU to use from CLI args and config.

    Priority: CLI ``--gpu`` flag > config ``gpu_id`` > default 0.

    Args:
        args_gpu: Value from ``argparse`` (may be ``None``).
        cfg_gpu_id: Value from the YAML config file.

    Returns:
        Resolved GPU index.
    """
    if args_gpu is not None:
        return int(args_gpu)
    return int(cfg_gpu_id)
