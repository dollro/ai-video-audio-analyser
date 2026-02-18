# CLAUDE.md — Developer Context

## Project overview

Two serverless Modal scripts for AI-powered media analysis:

- **`video-analyser.py`** — video understanding with Qwen VL/Omni models (visual + audio)
- **`audio-transcript.py`** — audio/video transcription with WhisperX (word timestamps + speaker diarization)

Neither script runs locally. All computation happens on Modal's cloud GPU instances.

### Infrastructure stack

| Component | video-analyser | audio-transcript |
|-----------|---------------|-----------------|
| Base image | `nvidia/cuda:12.8.0-devel-ubuntu22.04` | `modal.Image.debian_slim` |
| Python | 3.12 | 3.12 |
| CUDA | 12.8 | (bundled via PyTorch pip) |
| PyTorch | 2.7.0 (cu128) | 2.7.0 |
| flash-attn | 2.8.3 (compiled from source) | — |
| whisperx | — | 3.8.1 |
| Installer | `uv_pip_install` | `uv_pip_install` |

## How to run

```bash
# Video analysis (--video-url is required)
modal run video-analyser.py --video-url "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4"

# Audio + visual with Qwen3-Omni
modal run video-analyser.py --model qwen3-omni-30b-thinking --video-url "https://example.com/video.mp4"

# Transcription — accepts audio or video URLs
modal run audio-transcript.py --audio-url "https://example.com/audio.mp3"

# Hot-reload web endpoint during development
modal serve video-analyser.py

# Deploy persistent web API
modal deploy video-analyser.py
```

## Key files

| File | Purpose |
|---|---|
| `video-analyser.py` | Main video analysis app — Modal image, two analyzer classes, FastAPI web endpoint |
| `audio-transcript.py` | WhisperX transcription app — Modal image, single model class (accepts audio + video) |
| `.env.example` | Template for local env vars (MODAL tokens, HF_TOKEN) |
| `pyproject.toml` | Project metadata and dependencies (managed by uv) |

## Architecture

### `video-analyser.py`

Two Modal classes dispatched based on model series:

- **`QwenVLAnalyzer`** — visual-only, `AutoModelForVision2Seq`, flash attention for Qwen3
- **`QwenOmniAnalyzer`** — audio+visual, `Qwen3OmniMoeForConditionalGeneration`, talker disabled

Routing logic lives in `main()` and `fastapi_app()`: if `model_config["series"] == "qwen3-omni"` → `QwenOmniAnalyzer`, otherwise → `QwenVLAnalyzer`.

**Chunked processing** (`QwenOmniAnalyzer`): videos longer than `min_duration_for_chunking` (default 120s) are split into `chunk_duration`-second segments via `ffmpeg -c copy` (no re-encode), processed individually, then concatenated.

**Model cache** lives in a persistent Modal Volume at `/cache` (shared between both scripts).

### `audio-transcript.py`

Single `WhisperXModel` class with three pipeline stages:
1. `whisperx.load_model("large-v2")` → raw segments
2. `whisperx.load_align_model` + `whisperx.align` → word-level timestamps
3. `whisperx.diarize.DiarizationPipeline` → speaker labels (requires `HF_TOKEN` env var)

`@modal.concurrent(max_inputs=15)` allows 15 simultaneous transcriptions on one container.

## Adding a new model

Add an entry to the `MODELS` dict in `video-analyser.py`:

```python
"my-model-key": {
    "id": "Org/Model-Name",
    "gpu": "A100",
    "memory": 65536,
    "description": "Short description",
    "series": "qwen3",  # or "qwen2", "qwen3-omni"
},
```

The routing logic in `main()` and `fastapi_app()` will pick it up automatically.

## Secrets

- **Modal credentials**: `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in env, or via `modal setup`
- **HuggingFace token**: required only for diarization in `audio-transcript.py`
  - Create Modal secret: `modal secret create huggingface-secret HF_TOKEN=hf_...`
  - Or set `HF_TOKEN` env var before running

The `hf_secret` in `audio-transcript.py` is `required=False` — the script will raise a clear error at runtime if diarization is requested without a token.

## Modal conventions

- Always use `import modal` and qualified names (`modal.App()`, `modal.Image.debian_slim()`)
- Name Apps, Volumes, Secrets with **kebab-case** (e.g. `qwen-video-analyzer`, `whisper-cache`)
- Put heavy `import` statements inside functions/methods, not at module level — global scope runs locally too
- Dependencies belong in Image definitions attached to Functions, not in `pyproject.toml` (which is only for the local `modal` CLI env)
- GPU strings: `"A100"`, `"A100-80GB"`, `"H100"`, `"H100:8"`, or `["H100", "A100", "any"]` for fallbacks
- Docs: [modal.com/docs](https://modal.com/docs) | [Examples](https://modal.com/docs/examples) | [Full LLM reference](https://modal.com/llms-full.txt)

## Common gotchas

- **First run is slow** — model weights (~15–70 GB) are downloaded to the Modal Volume. Subsequent runs start in seconds.
- **`ffmpeg -c copy` chunk extraction** can produce slightly inaccurate cut points due to keyframe alignment. This is intentional (fast) and acceptable for analysis tasks.
- **`inputs.to(model.device).to(model.dtype)`** in `QwenOmniAnalyzer` is required — omitting `.to(model.dtype)` causes dtype mismatch errors with bfloat16 models.
- **`thinker_do_sample=False`** is hardcoded in Omni inference for reproducible results; `temperature`/`top_p`/`top_k` are **not valid** for Omni generate and will trigger a warning.
- **Omni `generate()` returns `str`** with recent transformers — code handles both `str` (use directly) and tensor (decode via `batch_decode`) return types. Do not use `thinker_return_dict_in_generate=True`.
- WhisperX `large-v2` uses `float16` compute type; change to `int8` if GPU memory is tight.
