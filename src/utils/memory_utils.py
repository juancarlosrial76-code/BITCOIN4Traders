"""
Centralized Memory Management Utilities
=======================================
Optimized for Google Colab and long-running trading loops.
Implements the "Saw Pattern" memory cleanup and explicit RAM release.
"""

import gc
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from loguru import logger


def load_memory_config() -> dict:
    """Loads memory management configuration."""
    cfg_path = Path("config/memory_management.yaml")
    if not cfg_path.exists():
        return {}  # Return empty dict if config file doesn't exist
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}  # Ensure None is replaced with empty dict


def cleanup_memory(force_cuda: bool = True):
    """
    Radically frees RAM and GPU memory.
    - Closes all Matplotlib figures
    - Forces Garbage Collection
    - Clears CUDA cache
    """
    # 1. Matplotlib cleanup (major cause of memory leaks in long training loops)
    plt.close("all")

    # 2. Explicit Python garbage collection (frees cyclic references)
    gc.collect()

    # 3. CUDA cleanup (only if a GPU is available and caller requests it)
    if force_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached but unused GPU memory blocks

    logger.debug("Memory cleanup performed")


def save_and_trim_data(
    df: pd.DataFrame,
    filename: str = "trade_history.csv",
    max_rows: int = 1000,
    drive_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    "Saw Pattern" Implementation:
    1. Saves the full dataframe to CSV (append mode) on Drive/Local
    2. Trims the dataframe in RAM to max_rows
    3. Frees memory explicitly
    """
    cfg = load_memory_config()
    # Prefer explicit drive_path; fall back to config file setting; then use "logs"
    target_dir = drive_path or cfg.get("drive_sync", {}).get("drive_log_path", "logs")

    # Ensure the target directory exists before writing
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, filename)

    # Append to existing CSV; write header only on first write
    header = not os.path.exists(full_path)
    df.to_csv(full_path, mode="a", header=header, index=False)

    # Keep only the most recent max_rows rows in memory (the "saw" cut)
    df_trimmed = df.tail(max_rows).copy()

    # Delete original reference so GC can collect it
    del df
    cleanup_memory(force_cuda=False)  # CPU-only cleanup here

    return df_trimmed


def get_ram_usage() -> float:
    """Returns current RAM usage in MB."""
    import psutil

    process = psutil.Process(os.getpid())  # Get the current process
    return process.memory_info().rss / 1024 / 1024  # Convert bytes → MB


def print_resource_report():
    """Prints current RAM and GPU usage."""
    ram = get_ram_usage()
    report = f"Resources: RAM: {ram:.1f} MB"

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # Bytes → MB
        report += f" | GPU: {gpu_mem:.1f} MB"

    logger.info(report)
