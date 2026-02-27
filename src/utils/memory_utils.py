"""
Centralized Memory Management Utilities
======================================
Optimized memory management system for long-running trading loops.

This module provides comprehensive memory management utilities designed
for Google Colab environments and production trading systems with
extended runtime requirements.

FEATURES:
--------
1. MEMORY CLEANUP
   - Matplotlib figure cleanup
   - Python garbage collection
   - CUDA cache management

2. DATA PERSISTENCE ("Saw Pattern")
   - Save data to disk in chunks
   - Trim in-memory data to max rows
   - Automatic cleanup

3. RESOURCE MONITORING
   - RAM usage tracking
   - GPU memory monitoring
   - Resource usage reports

WHY MEMORY MANAGEMENT MATTERS:
----------------------------
- Long-running trading systems accumulate memory
- Matplotlib figures leak memory
- GPU memory fragmentation
- Colab has limited RAM

THE "SAW PATTERN":
-----------------
Visualization of memory usage over time:

    RAM Usage
        ^
        |    /‾‾‾\     /‾‾‾\     /‾‾‾\
        |   /    \   /    \   /    \
        |  /      \ /      \ /      \
        | /        \/        \/        \
        +----------------------------------> Time

Each "tooth" represents:
1. Save accumulated data to disk
2. Trim in-memory data to max_rows
3. Free memory explicitly

This prevents unbounded memory growth while preserving data.

Usage:
    from src.utils.memory_utils import (
        cleanup_memory,
        save_and_trim_data,
        get_ram_usage,
        print_resource_report
    )
    
    # Regular cleanup
    cleanup_memory()
    
    # Save and trim
    df = save_and_trim_data(df, "trades.csv", max_rows=1000)
    
    # Monitor
    print_resource_report()

Author: BITCOIN4Traders Team
License: Proprietary - Internal Use Only
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
    """
    Load memory management configuration from YAML file.

    Reads configuration from config/memory_management.yaml if it exists.
    Returns empty dict if file doesn't exist.

    Returns:
        Dictionary with memory management settings

    Example:
        >>> cfg = load_memory_config()
        >>> print(cfg.get('drive_sync', {}))
    """
    cfg_path = Path("config/memory_management.yaml")
    if not cfg_path.exists():
        return {}  # Return empty dict if config file doesn't exist
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}  # Ensure None is replaced with empty dict


def cleanup_memory(force_cuda: bool = True):
    """
    Radically free RAM and GPU memory.

    Performs three cleanup operations:
    1. Closes all Matplotlib figures (major memory leak source)
    2. Forces Python garbage collection
    3. Clears CUDA cache (if GPU available and requested)

    Args:
        force_cuda: Whether to clear CUDA cache (default: True)

    Note:
        Should be called periodically in long-running loops
        to prevent memory accumulation.

    Example:
        >>> # Every 100 iterations
        >>> if iteration % 100 == 0:
        ...     cleanup_memory()
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
    "Saw Pattern" Implementation for memory-efficient data handling.

    This function implements the saw pattern for memory management:
    1. Saves the full DataFrame to CSV (append mode)
    2. Trims the DataFrame to most recent max_rows
    3. Explicitly frees memory from old data

    This prevents unbounded memory growth in long-running systems
    while preserving all historical data on disk.

    Args:
        df: DataFrame to save and trim
        filename: CSV filename for persistence
        max_rows: Maximum rows to keep in memory after trim
        drive_path: Custom path for saving (optional)

    Returns:
        Trimmed DataFrame with max_rows

    Example:
        >>> # Process and save large dataset
        >>> for batch in batches:
        ...     df = process_data(batch)
        ...     df = save_and_trim_data(df, "trades.csv", max_rows=1000)
        ...     # df now only keeps last 1000 rows in memory
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
    """
    Get current process RAM usage.

    Returns the RAM usage of the current process in megabytes.

    Returns:
        RAM usage in MB

    Example:
        >>> ram = get_ram_usage()
        >>> print(f"Using {ram:.1f} MB")
    """
    import psutil

    process = psutil.Process(os.getpid())  # Get the current process
    return process.memory_info().rss / 1024 / 1024  # Convert bytes → MB


def print_resource_report():
    """
    Print current RAM and GPU memory usage.

    Outputs a formatted log message with:
    - RAM usage in MB
    - GPU memory usage (if CUDA available) in MB

    Useful for monitoring resource consumption in training loops.

    Example:
        >>> # Every 10 iterations
        >>> if iteration % 10 == 0:
        ...     print_resource_report()

        # Output: Resources: RAM: 245.3 MB | GPU: 1024.5 MB
    """
    ram = get_ram_usage()
    report = f"Resources: RAM: {ram:.1f} MB"

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # Bytes → MB
        report += f" | GPU: {gpu_mem:.1f} MB"

    logger.info(report)
