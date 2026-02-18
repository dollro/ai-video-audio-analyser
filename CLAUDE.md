# CLAUDE.md — Developer Context

## Project overview

Two serverless Modal scripts for AI-powered media analysis:

- **`video-analyser.py`** — video understanding with Qwen VL/Omni models (visual + audio)
- **`audio-transcript.py`** — audio/video transcription with WhisperX (word timestamps + speaker diarization)

Neither script runs locally. All computation happens on Modal's cloud GPU instances.

## How to run

```bash
# Video analysis (default: Qwen3-VL 8B on a test video)
modal run video-analyser.py

# Transcription — accepts audio or video URLs
modal run audio-transcript.py --audio-url "https://example.com/audio.mp3"

# Deploy web API
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

### `analyze_video.py`

Two Modal classes dispatched based on model series:

- **`QwenVLAnalyzer`** — visual-only, `AutoModelForVision2Seq`, flash attention for Qwen3
- **`QwenOmniAnalyzer`** — audio+visual, `Qwen3OmniMoeForConditionalGeneration`, talker disabled

Routing logic lives in `main()` and `fastapi_app()`: if `model_config["series"] == "qwen3-omni"` → `QwenOmniAnalyzer`, otherwise → `QwenVLAnalyzer`.

**Chunked processing** (`QwenOmniAnalyzer`): videos longer than `min_duration_for_chunking` (default 120s) are split into `chunk_duration`-second segments via `ffmpeg -c copy` (no re-encode), processed individually, then concatenated.

**Model cache** lives in a persistent Modal Volume at `/cache` (shared between both scripts).

### `transcribe.py`

Single `WhisperXModel` class with three pipeline stages:
1. `whisperx.load_model("large-v2")` → raw segments
2. `whisperx.load_align_model` + `whisperx.align` → word-level timestamps
3. `whisperx.diarize.DiarizationPipeline` → speaker labels (requires `HF_TOKEN` env var)

`@modal.concurrent(max_inputs=15)` allows 15 simultaneous transcriptions on one container.

## Adding a new model

Add an entry to the `MODELS` dict in `analyze_video.py`:

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
- **HuggingFace token**: required only for diarization in `transcribe.py`
  - Create Modal secret: `modal secret create huggingface-secret HF_TOKEN=hf_...`
  - Or set `HF_TOKEN` env var before running

The `hf_secret` in `transcribe.py` is `required=False` — the script will raise a clear error at runtime if diarization is requested without a token.

## Common gotchas

- **First run is slow** — model weights (~15–70 GB) are downloaded to the Modal Volume. Subsequent runs start in seconds.
- **`ffmpeg -c copy` chunk extraction** can produce slightly inaccurate cut points due to keyframe alignment. This is intentional (fast) and acceptable for analysis tasks.
- **`inputs.to(model.device).to(model.dtype)`** in `QwenOmniAnalyzer` is required — omitting `.to(model.dtype)` causes dtype mismatch errors with bfloat16 models.
- **`thinker_do_sample=False`** is hardcoded in Omni inference for reproducible results; the `temperature`/`top_p` params are accepted by the API but not passed to the thinker.
- WhisperX `large-v2` uses `float16` compute type; change to `int8` if GPU memory is tight.
