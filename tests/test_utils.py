"""Tests for utility modules."""

import os
import tempfile
import pytest
import torch
import numpy as np

from synctalk.utils.device import get_device, to_device
from synctalk.utils.io import ensure_dir, safe_remove, read_landmarks, get_crop_region


class TestDevice:
    def test_auto_device(self):
        dev = get_device("auto")
        assert isinstance(dev, torch.device)

    def test_cpu_device(self):
        dev = get_device("cpu")
        assert dev == torch.device("cpu")

    def test_to_device(self):
        t = torch.randn(3, 3)
        result = to_device(t, torch.device("cpu"))
        assert result.device == torch.device("cpu")


class TestIO:
    def test_ensure_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "a", "b", "c")
            result = ensure_dir(new_dir)
            assert os.path.isdir(new_dir)
            assert result.exists()

    def test_safe_remove(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        assert os.path.exists(path)
        safe_remove(path)
        assert not os.path.exists(path)

    def test_safe_remove_nonexistent(self):
        safe_remove("/nonexistent/path/file.txt")

    def test_read_landmarks(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lms", delete=False) as f:
            for i in range(68):
                f.write(f"{100 + i} {200 + i}\n")
            path = f.name

        try:
            lms = read_landmarks(path)
            assert lms.shape == (68, 2)
            assert lms[0, 0] == 100
            assert lms[0, 1] == 200
        finally:
            os.unlink(path)
