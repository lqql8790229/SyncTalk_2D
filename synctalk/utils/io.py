"""File I/O and subprocess utilities."""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with proper error handling."""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        logger.error(f"Command failed: {cmd}\nstderr: {result.stderr}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}\n{result.stderr}")
    return result


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path: str | Path) -> None:
    """Safely remove a file."""
    path = Path(path)
    if path.exists():
        path.unlink()


def read_landmarks(lms_path: str | Path) -> list:
    """Read landmark file and return list of (x, y) coordinates."""
    import numpy as np
    lms_list = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    return np.array(lms_list, dtype=np.int32)


def get_crop_region(lms):
    """Calculate crop region from landmarks."""
    import numpy as np
    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    return int(xmin), int(ymin), int(xmax), int(ymax), int(width)
