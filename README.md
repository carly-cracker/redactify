# Redactify

Simple video redaction tool that detects people, faces and other objects using YOLO and applies redaction (blur / pixelate / black box).

This repository contains two UIs and a backend processing pipeline:

- `redactify/` — Python pieces (model downloader and `redact video.py and app.py` for processing).
- `models/` — place model weights (e.g. `yolov9c.pt`, `yolov8n-face.pt`) here.

This README explains how to set up the project, run it locally, and common troubleshooting steps.

---
## Quick TL;DR

- Use the Streamlit UI for local experiments (single-machine):
  - `pip install -r requirements.txt` (see note about `torch` below)
  - `streamlit run app.py`
- Use the React frontend with a backend API for a production-like setup.
- Ensure model weights live in `./models/` and that `ultralytics`/`torch` are installed in your Python venv.

---
## Prerequisites

- Python 3.8+ and `virtualenv` or equivalent.
- Node.js (for the React frontend) if you want to run the SPA in `redactify-front/`.
- Enough disk space for model weights (hundreds of MB each).
- (Optional) CUDA-enabled GPU + matching `torch` wheel for much faster inference.

---
## Backend (Python) setup

1. Create and activate a virtual environment (example):

```bash
cd /path/to/redactify
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
# project-root requirements (Streamlit + helpers)
pip install -r requirements.txt

# If you will use the backend folder requirements instead:
# cd redactify-back
# ./.venv/bin/python3 -m pip install -r requirements.txt
```

3. Install PyTorch (CPU) if not present (choose CPU or appropriate CUDA build):

```bash
# CPU-only wheel (example)
python -m pip install "torch" --index-url https://download.pytorch.org/whl/cpu
```

Notes:
- `ultralytics` depends on `torch`. Installing `torch` first (with the correct CUDA variant for your machine) avoids surprises.
- If you plan to use GPU, install the matching CUDA `torch` wheel recommended by PyTorch: https://pytorch.org/get-started/locally/

---
## Models

Put model files under `./models/` (project root). Example expected filenames include:

- `models/yolov9c.pt` (general detector)
- `models/yolov8n-face.pt` (specialized face detector, optional)

You can download models manually or use the `download_models.py` script in `redactify-back/`:

```bash
# from project root or redactify-back/
python3 download_models.py
ls -l models
```

If the script cannot download automatically (network or permission issues), place the files manually in `models/`.

---
## Running Streamlit (quick local UI)

Streamlit runs the bundled `app.py` which calls the local processing function. This is easiest for experiments.

```bash
# Activate venv first
source .venv/bin/activate
streamlit run app.py
```

Notes:
- Streamlit UI runs processing synchronously by default (the UI waits while `process_video` runs). For long videos prefer the job-based API (see below) or process a short preview with `frame_skip` / `max_frames` to validate config.




---
## Recommended way forward (developer notes)

1. Short-term: keep using Streamlit for local testing. Use `frame_skip` and `max_frames` settings for quick iteration.
2. Medium-term: implement a job queue (RQ/Celery or a simple thread-based manager) and add two endpoints:
   - `POST /api/redact` — accepts video, returns `job_id`
   - `GET /api/status/<job_id>` — returns job status and optional progress/logs
   Frontend can upload and poll status or subscribe to SSE/WebSocket for live logs.
3. Long-term: use GPU inference, batching, or optimized runtimes (ONNX / TensorRT) for production-level throughput.

---
## Troubleshooting

- "Model file not found": verify `models/` exists and contains the `.pt` files. Example:

```bash
ls -l models
# if empty, download or move weights into this folder
```

- `ModuleNotFoundError` importing `redact_video` from `app.py`: run Streamlit from project root so local module imports work, or add an `__init__.py` and use package imports.

- `ultralytics` / `torch` errors: ensure `torch` is installed and the CUDA/CPU build matches your environment. Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

- Slow processing / UI hangs: enable `frame_skip` and smaller `INFERENCE_MAX_DIM`, or run on GPU. Prefer job queue for large inputs.

---
## Development tips

- Add `logging` instead of `print` to capture structured logs in services.
- Add a `tests/` folder with a small sample clip and integration test using `max_frames` to catch regressions.
- Keep model filenames and the downloader script in sync.

---
## Contributing

PRs welcome. Keep changes focused, add tests for processing utilities when possible, and describe performance impacts for model or pipeline changes.

---
## License

This repository does not include a license file. Add an appropriate license (`MIT`, `Apache-2.0`, etc.) if you intend to share this publicly.

---
If you want, I can now:
- Add a minimal HTTP job API (Flask or FastAPI) and a simple React integration for non-blocking uploads, or
- Convert `print()` to `logging` throughout the backend and run a small timing test to measure per-frame inference time.

Tell me which you'd prefer as the next step.