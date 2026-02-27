"""ONNX model export utilities."""

import copy
import logging
from pathlib import Path

import torch
import numpy as np

from .unet import UNet
from ..configs.base import ModelConfig

logger = logging.getLogger(__name__)


def export_unet_onnx(model: UNet, config: ModelConfig,
                     output_path: str, opset_version: int = 11,
                     verify: bool = True) -> str:
    """Export UNet model to ONNX format.

    Args:
        model: Trained UNet model.
        config: Model configuration.
        output_path: Output ONNX file path.
        opset_version: ONNX opset version.
        verify: Whether to verify exported model with ONNX Runtime.

    Returns:
        Path to exported ONNX file.
    """
    model = copy.deepcopy(model).eval().cpu()
    output_path = str(Path(output_path).with_suffix(".onnx"))

    inner_size = config.inner_size
    feat_shape = config.audio_feat_shape

    dummy_img = torch.zeros(1, 6, inner_size, inner_size)
    dummy_audio = torch.zeros(1, *feat_shape)

    logger.info(f"Exporting UNet to ONNX: {output_path}")
    logger.info(f"  Input image: {list(dummy_img.shape)}")
    logger.info(f"  Input audio: {list(dummy_audio.shape)}")

    with torch.no_grad():
        torch_out = model(dummy_img, dummy_audio)

    torch.onnx.export(
        model,
        (dummy_img, dummy_audio),
        output_path,
        input_names=["image", "audio"],
        output_names=["output"],
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )

    logger.info(f"Exported: {output_path}")

    if verify:
        _verify_onnx(output_path, dummy_img, dummy_audio, torch_out)

    return output_path


def _verify_onnx(onnx_path: str, dummy_img, dummy_audio, torch_out):
    """Verify ONNX model matches PyTorch output."""
    try:
        import onnx
        import onnxruntime

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        providers = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        ort_inputs = {
            session.get_inputs()[0].name: dummy_img.numpy(),
            session.get_inputs()[1].name: dummy_audio.numpy(),
        }
        ort_out = session.run(None, ort_inputs)

        np.testing.assert_allclose(
            torch_out[0].numpy(), ort_out[0][0],
            rtol=1e-3, atol=1e-5,
        )
        logger.info("ONNX verification passed!")
    except ImportError:
        logger.warning("onnx/onnxruntime not available, skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")


def get_model_info(model: torch.nn.Module) -> dict:
    """Get model size and parameter info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_size_mb": round(total_size_mb, 2),
    }
