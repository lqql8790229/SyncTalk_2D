FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY synctalk/ synctalk/

RUN pip install --no-cache-dir -e ".[gpu]"

COPY model/ model/
COPY data_utils/scrfd_2.5g_kps.onnx data_utils/scrfd_2.5g_kps.onnx
COPY data_utils/checkpoint_epoch_335.pth.tar data_utils/checkpoint_epoch_335.pth.tar
COPY data_utils/mean_face.txt data_utils/mean_face.txt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "synctalk.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
