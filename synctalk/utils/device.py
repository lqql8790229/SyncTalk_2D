"""Unified device management."""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        device_str: "auto", "cuda", "cuda:0", "cpu", "mps"
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def to_device(tensor_or_model, device: torch.device):
    """Move tensor or model to device with proper handling."""
    return tensor_or_model.to(device)
