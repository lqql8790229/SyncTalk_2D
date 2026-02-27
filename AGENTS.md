# AGENTS.md

## Cursor Cloud specific instructions

### Overview
SyncTalk_2D is a Python ML pipeline for 2D lip-sync video generation. It takes a source video + audio and generates a lip-synced output. The codebase has two layers:
- **Legacy code** (root-level `*.py`): Original scripts with hardcoded `.cuda()` and duplicate 160/328px files.
- **`synctalk/` package**: Refactored commercial-grade package with unified models, configs, API, and CLI.

### Environment
- **Python 3.10** required (installed via `deadsnakes` PPA: `python3.10`).
- Virtual environment at `/workspace/.venv` (activate: `source /workspace/.venv/bin/activate`).
- **No GPU** in the Cloud Agent VM. PyTorch CPU is installed. All model imports, weight loading, audio feature extraction, and UNet forward passes work on CPU.
- `ffmpeg` is pre-installed system-wide.

### Key Gotchas
- **No `requirements.txt`** in this repo. Dependencies are listed in `README.md`. The update script installs them via pip.
- **`numpy==1.23.5`** is pinned. Must install after other packages since newer opencv/scipy may pull a newer numpy first.
- **`opencv-python==4.8.1.78`** is used for numpy 1.23.5 compatibility. Newer opencv requires numpy>=2.
- **`data_utils/ave/`** scripts (e.g., `test_w2l_audio.py`, `audio.py`) use **implicit relative imports** (`from hparams import ...`, `import audio`). They are designed to run from within that directory (called via `data_utils/process.py`). Do not import them as standard packages from the workspace root.
- **`data_utils/pfld_mobileone.py`** similarly imports `base_module` relatively. It runs correctly when invoked from the `data_utils/` directory during the preprocessing pipeline.
- Some code paths (e.g., `inference_328.py` line 68, `data_utils/ave/test_w2l_audio.py` line 118) hardcode `.cuda()`. These will fail on CPU. The `inference_328.py` line 37 does check `torch.cuda.is_available()` for the AudioEncoder device, but the UNet loading uses `.cuda()` directly. Training and full inference require a GPU.

### Running (new synctalk package)
- **Install**: `pip install -e .` (installs the `synctalk` package + all dependencies from `pyproject.toml`).
- **CLI**: `python -m synctalk.cli --help` — unified entry point for preprocess/train/inference/serve.
- **API server**: `python -m synctalk.cli serve --port 8000` — starts FastAPI server at `http://localhost:8000/docs`.
- **Lint**: `python -m py_compile <file.py>` on individual files.
- **Smoke test** (CPU, no GPU needed):
  ```bash
  source /workspace/.venv/bin/activate && cd /workspace
  python -c "
  from synctalk.models import UNet, AudioEncoder
  from synctalk.data.audio import AudDataset
  from torch.utils.data import DataLoader; import torch
  ae = AudioEncoder().eval(); ae.load_pretrained('model/checkpoints/audio_visual_encoder.pth')
  ds = AudDataset('demo/talk_hb.wav'); dl = DataLoader(ds, batch_size=64)
  outs = torch.cat([ae(m) for m in dl]).cpu(); print(f'Audio features: {outs.shape}')
  unet = UNet(6, 'ave', n_down_layers=5)
  pred = unet(torch.randn(1,6,320,320), torch.randn(1,32,16,16))
  print(f'UNet forward: {pred.shape}')
  "
  ```
- **Training** (requires GPU): `python -m synctalk.cli train --stage full --dataset_dir ./dataset/May --use_syncnet`
- **Inference** (requires GPU): `python -m synctalk.cli inference --name May --audio_path demo/talk_hb.wav`

### Running (legacy code)
- **Training** (requires GPU): `bash training_328.sh <name> <gpu_id>` — see `README.md`.
- **Inference** (requires GPU): `python inference_328.py --name <name> --audio_path <wav>` — see `README.md`.
