"""Python SDK client for SyncTalk API.

Usage:
    from synctalk.sdk import SyncTalkClient

    client = SyncTalkClient("http://localhost:8000", api_key="your-key")

    # Check service health
    health = client.health()

    # Submit inference task
    task = client.inference("May", "audio.wav")
    result = client.wait_for_task(task["task_id"])

    # Download generated video
    client.download_video(task["task_id"], "output.mp4")
"""

import time
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class SyncTalkClient:
    """Python SDK for SyncTalk API.

    Args:
        base_url: API server URL (e.g., "http://localhost:8000").
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, **kwargs):
        resp = self.session.get(self._url(path), timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json=None, **kwargs):
        resp = self.session.post(self._url(path), json=json, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Check API server health."""
        return self._get("/health")

    def inference(self, name: str, audio_path: str,
                  checkpoint_path: Optional[str] = None,
                  resolution: int = 328, asr_mode: str = "ave") -> dict:
        """Submit an inference task.

        Args:
            name: Dataset/project name.
            audio_path: Path to audio WAV file (on server).
            checkpoint_path: Optional model checkpoint path.
            resolution: Resolution (160 or 328).
            asr_mode: ASR mode ("ave", "hubert", "wenet").

        Returns:
            Task submission response with task_id.
        """
        payload = {
            "name": name,
            "audio_path": audio_path,
            "resolution": resolution,
            "asr_mode": asr_mode,
        }
        if checkpoint_path:
            payload["checkpoint_path"] = checkpoint_path
        return self._post("/api/v1/inference", json=payload)

    def train(self, name: str, resolution: int = 328, epochs: int = 100,
              batch_size: int = 8, asr_mode: str = "ave",
              use_syncnet: bool = True) -> dict:
        """Submit a training task.

        Returns:
            Task submission response with task_id.
        """
        return self._post("/api/v1/train", json={
            "name": name,
            "resolution": resolution,
            "epochs": epochs,
            "batch_size": batch_size,
            "asr_mode": asr_mode,
            "use_syncnet": use_syncnet,
        })

    def get_task(self, task_id: str) -> dict:
        """Get task status."""
        return self._get(f"/api/v1/tasks/{task_id}")

    def wait_for_task(self, task_id: str, poll_interval: float = 5.0,
                      timeout: float = 3600) -> dict:
        """Wait for a task to complete.

        Args:
            task_id: Task ID to monitor.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds.

        Returns:
            Final task status.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
            RuntimeError: If task fails.
        """
        start = time.time()
        while True:
            status = self.get_task(task_id)

            if status["status"] == "completed":
                logger.info(f"Task {task_id} completed")
                return status

            if status["status"] == "failed":
                raise RuntimeError(
                    f"Task {task_id} failed: {status.get('error', 'Unknown error')}"
                )

            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Task {task_id} timed out after {timeout}s "
                    f"(status: {status['status']})"
                )

            logger.debug(f"Task {task_id}: {status['status']} ({elapsed:.0f}s)")
            time.sleep(poll_interval)

    def download_video(self, task_id: str, output_path: str) -> str:
        """Download generated video from completed inference task.

        Args:
            task_id: Completed inference task ID.
            output_path: Local path to save the video.

        Returns:
            Path to downloaded file.
        """
        resp = self.session.get(
            self._url(f"/api/v1/inference/{task_id}/video"),
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {output_path}")
        return output_path

    def list_projects(self) -> list:
        """List available projects."""
        return self._get("/api/v1/projects")

    def model_info(self) -> dict:
        """Get model architecture information."""
        return self._get("/api/v1/models/info")
