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
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def cleanup_memory(force_cuda: bool = True):
    """
    Radically frees RAM and GPU memory.
    - Closes all Matplotlib figures
    - Forces Garbage Collection
    - Clears CUDA cache
    """
    # 1. Matplotlib cleanup (Major cause of memory leaks)
    plt.close("all")
    
    # 2. explicit GC
    gc.collect()
    
    # 3. CUDA cleanup
    if force_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.debug("Memory cleanup performed")


def save_and_trim_data(
    df: pd.DataFrame, 
    filename: str = "trade_history.csv", 
    max_rows: int = 1000,
    drive_path: Optional[str] = None
) -> pd.DataFrame:
    """
    "Saw Pattern" Implementation:
    1. Saves the full dataframe to CSV (append mode) on Drive/Local
    2. Trims the dataframe in RAM to max_rows
    3. Frees memory explicitly
    """
    cfg = load_memory_config()
    target_dir = drive_path or cfg.get("drive_sync", {}).get("drive_log_path", "logs")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, filename)
    
    # Save to disk (append mode)
    header = not os.path.exists(full_path)
    df.to_csv(full_path, mode='a', header=header, index=False)
    
    # Trim RAM
    df_trimmed = df.tail(max_rows).copy()
    
    # Cleanup old references
    del df
    cleanup_memory(force_cuda=False)
    
    return df_trimmed


def get_ram_usage() -> float:
    """Returns current RAM usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_resource_report():
    """Prints current RAM and GPU usage."""
    ram = get_ram_usage()
    report = f"Resources: RAM: {ram:.1f} MB"
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        report += f" | GPU: {gpu_mem:.1f} MB"
        
    logger.info(report)
