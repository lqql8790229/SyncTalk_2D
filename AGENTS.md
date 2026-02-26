# AGENTS.md

## Cursor Cloud specific instructions

### Overview
SyncTalk_2D is a Python ML pipeline for 2D lip-sync video generation. It takes a source video + audio and generates a lip-synced output. It is a CLI-based training/inference tool (no web server, no database).

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

### Running
- **Lint**: No linter is configured in this project. You can run `python -m py_compile <file.py>` on individual files.
- **Tests**: No automated test suite exists. Validate by importing modules and running forward passes (see hello-world smoke test below).
- **Hello-world smoke test** (CPU, no GPU needed):
  ```bash
  source /workspace/.venv/bin/activate
  cd /workspace
  python -c "
  from utils import AudioEncoder, AudDataset; from torch.utils.data import DataLoader; import torch
  model = AudioEncoder().eval()
  model.load_state_dict({f'audio_encoder.{k}': v for k, v in torch.load('model/checkpoints/audio_visual_encoder.pth', map_location='cpu').items()})
  ds = AudDataset('demo/talk_hb.wav'); dl = DataLoader(ds, batch_size=64)
  outs = torch.cat([model(m) for m in dl]).cpu(); print(f'Audio features: {outs.shape}')
  "
  ```
- **Training** (requires GPU): `bash training_328.sh <name> <gpu_id>` — see `README.md`.
- **Inference** (requires GPU): `python inference_328.py --name <name> --audio_path <wav>` — see `README.md`.
