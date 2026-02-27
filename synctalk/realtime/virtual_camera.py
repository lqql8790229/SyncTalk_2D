"""Virtual camera output for streaming lip-synced video.

Uses pyvirtualcam to create a virtual camera device that can be
used in any video conferencing app (Zoom, Teams, OBS, etc.).

Supported backends:
- Windows: OBS Virtual Camera (OBS must be installed)
- macOS: OBS Virtual Camera
- Linux: v4l2loopback
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class VirtualCameraOutput:
    """Virtual camera output stream.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        fps: Output frame rate.
        backend: Backend to use (None for auto-detect).
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 fps: int = 25, backend: Optional[str] = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._cam = None

    def start(self):
        """Start the virtual camera."""
        try:
            import pyvirtualcam
        except ImportError:
            raise ImportError(
                "pyvirtualcam not installed. Install it with:\n"
                "  pip install pyvirtualcam\n"
                "Windows: Also install OBS Studio for the virtual camera driver.\n"
                "Linux: Also install v4l2loopback-dkms."
            )

        kwargs = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
        if self.backend:
            kwargs["backend"] = self.backend

        self._cam = pyvirtualcam.Camera(**kwargs)
        logger.info(
            f"Virtual camera started: {self._cam.device} "
            f"({self.width}x{self.height} @ {self.fps}fps)"
        )

    def send_frame(self, frame_bgr: np.ndarray):
        """Send a BGR frame to the virtual camera.

        Args:
            frame_bgr: BGR image (OpenCV format). Will be resized and
                       converted to RGB automatically.
        """
        if self._cam is None:
            raise RuntimeError("Virtual camera not started. Call start() first.")

        import cv2

        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._cam.send(frame_rgb)

    def stop(self):
        """Stop the virtual camera."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None
            logger.info("Virtual camera stopped")

    @property
    def is_running(self) -> bool:
        return self._cam is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class PreviewWindow:
    """Simple OpenCV preview window for monitoring output.

    Args:
        window_name: Window title.
    """

    def __init__(self, window_name: str = "SyncTalk Preview"):
        self.window_name = window_name
        self._open = False

    def show(self, frame_bgr: np.ndarray):
        """Display a frame in the preview window."""
        import cv2
        cv2.imshow(self.window_name, frame_bgr)
        self._open = True
        return cv2.waitKey(1) & 0xFF

    def close(self):
        """Close the preview window."""
        if self._open:
            import cv2
            cv2.destroyWindow(self.window_name)
            self._open = False
