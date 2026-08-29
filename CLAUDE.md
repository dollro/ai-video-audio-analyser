# CLAUDE.md — Developer Context

## Project overview

Two serverless Modal scripts for AI-powered media analysis:

- **`video-analyser.py`** — video understanding with Qwen VL/Omni models (visual + audio)
- **`audio-transcript.py`** — audio/video transcription with WhisperX (word timestamps + speaker diarization)

Neither script runs locally. All computation happens on Modal's cloud GPU instances.

### Infrastructure stack

| Component | video-analyser | audio-transcript |
|-----------|---------------|-----------------|
| Base image | `nvidia/cuda:12.9.0-devel-ubuntu22.04` | `modal.Image.debian_slim` |
| Python | 3.12 | 3.12 |
| CUDA | 12.9 | (bundled via PyTorch pip) |
| PyTorch | 2.13.0+cu130 (resolved by vLLM 0.27.1, observed at runtime — not pinned in the image) | 2.8.0 (cu128) |
| vLLM | 0.27.1 | — |
| whisperx | — | 3.8.6 |
| Installer | `uv_pip_install` | `uv_pip_install` |

## How to run

```bash
# Video analysis (--video-url is required)
modal run video-analyser.py --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"

# Audio + visual with Qwen3-Omni
modal run video-analyser.py --model qwen3-omni-30b-instruct --video-url "https://example.com/video.mp4"

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

Two Modal classes dispatched based on model series, both using **vLLM** for inference:

- **`QwenVLAnalyzer`** — visual-only, vLLM `LLM()` with `process_vision_info` for multimodal data
- **`QwenOmniAnalyzer`** — audio+visual, vLLM `LLM()` with tensor parallelism (tp=2) on 2x A100-80GB

Routing logic lives in `main()` and `fastapi_app()`: if `model_config["series"] == "qwen3-omni"` → `QwenOmniAnalyzer`, otherwise → `QwenVLAnalyzer` — but only when `tensor_parallel_size == 1`; both entry points raise before dispatch if a non-Omni model declares `tensor_parallel_size > 1`.

**Tensor parallelism**: each model in `MODELS` has a `tensor_parallel_size` field, but only two values are actually usable today — `qwen3-omni` models run at tp=2 (2 GPUs) and single-GPU VL models run at tp=1. `qwen3-vl-235b` declares `tensor_parallel_size: 8`, but `QwenVLAnalyzer` is hardcoded to a single A100, and `main()`/`fastapi_app()` reject any non-Omni model with `tensor_parallel_size > 1` ("Multi-GPU VL support is not yet implemented") — there is no working 8-GPU path for it.

**Batched chunk processing** (`QwenOmniAnalyzer`): videos longer than `min_duration_for_chunking` (default 120s) are split into segments via `ffmpeg -c copy`, then all chunks are submitted in a single `llm.generate()` call — vLLM's continuous batching processes them concurrently across GPUs.

**Model cache**: each script mounts its own persistent Modal Volume at `/cache` — `qwen3-vl-cache` (`video-analyser.py:54`) for `video-analyser.py`, `whisper-cache` (`audio-transcript.py:32`) for `audio-transcript.py`. They share the mount path, not the underlying Volume — cached weights are not shared between the two scripts.

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
    "tensor_parallel_size": 1,  # number of GPUs for vLLM tensor parallelism
    "max_model_len": 32768,
},
```

The `"gpu"` and `"memory"` fields are display-only — they show up in printed/error text and the `/models` endpoint, but they don't provision anything. Real hardware is fixed by the `@app.cls(gpu=...)` decorator: `QwenVLAnalyzer` is always `gpu="A100"` (`video-analyser.py:155-156`) and `QwenOmniAnalyzer` is always `gpu="A100-80GB:2"` (`:377-378`). Changing a MODELS entry's `"gpu"` value changes nothing about the container it runs on.

For a `tensor_parallel_size` of 1, the routing logic in `main()` and `fastapi_app()` picks up a new `qwen2`/`qwen3` entry automatically and dispatches it to `QwenVLAnalyzer`. A `tensor_parallel_size > 1` on a non-`qwen3-omni` entry is rejected at runtime instead — `QwenVLAnalyzer` only supports single-GPU models, and multi-GPU VL support isn't implemented (see `qwen3-vl-235b` above).

## Secrets

- **Modal CLI credentials** (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`): must be in the process environment or in `~/.modal.toml` (via `modal setup`). The `modal` CLI does not read `.env` — if your tokens only live there, export them first: `set -a && . ./.env && set +a`.
- **HuggingFace token**: both `video-analyser.py:152` and `audio-transcript.py:36` load it via `hf_secret = modal.Secret.from_dotenv()`. This reads the local `.env` **file** at run/deploy time and ships its contents into the container — it does not read the parent process environment and does not consult a named Modal secret. `modal secret create huggingface-secret HF_TOKEN=...` has no effect on these scripts; `HF_TOKEN` must be set in `.env`.

`Secret.from_dotenv()` has no `required` parameter — a missing `.env` just produces an empty secret. `audio-transcript.py` raises its own `ValueError`/`RuntimeError` at runtime if diarization is requested without a usable `HF_TOKEN`.

## Modal conventions

- Always use `import modal` and qualified names (`modal.App()`, `modal.Image.debian_slim()`)
- Name Apps, Volumes, Secrets with **kebab-case** (e.g. `qwen-video-analyzer`, `whisper-cache`)
- Put heavy `import` statements inside functions/methods, not at module level — global scope runs locally too
- Dependencies belong in Image definitions attached to Functions, not in `pyproject.toml` (which is only for the local `modal` CLI env)
- GPU strings: `"A100"`, `"A100-80GB"`, `"H100"`, `"H100:8"`, or `["H100", "A100", "any"]` for fallbacks
- Docs: [modal.com/docs](https://modal.com/docs) | [Examples](https://modal.com/docs/examples) | [Full LLM reference](https://modal.com/llms-full.txt)

## Common gotchas

- **First run is slow** — model weights (~15–70 GB) are downloaded to the Modal Volume once. Subsequent runs skip the download, but every container start still pays for vLLM engine init (see `enforce_eager` below) — "faster" doesn't mean "instant."
- **`ffmpeg -c copy` chunk extraction** can produce slightly inaccurate cut points due to keyframe alignment. This is intentional (fast) and acceptable for analysis tasks.
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** is required globally — vLLM's multiprocess executor forks by default, but pynvml calls `cuInit()` before fork which breaks CUDA in child workers. Spawn creates fresh processes and avoids this.
- **`max_num_seqs=8`** limits concurrent batch processing in vLLM. For chunked videos, all chunks are submitted in one `llm.generate()` call but vLLM queues beyond 8.
- **Multi-modal data dict** — vLLM expects `{"prompt": ..., "multi_modal_data": {"video": ..., "audio": ...}}` format. The processor's `process_vision_info` / `process_mm_info` output feeds directly into `multi_modal_data`.
- **`mm_processor_kwargs`** — required in the vLLM input dict for VL models. `process_vision_info(..., return_video_kwargs=True)` returns a third value (`video_kwargs`) that must be passed through. transformers 5.x validates processor kwargs with strict `huggingface_hub` dataclasses, which broke this in two ways after the upgrade: on the Qwen2 (BC) path, qwen-vl-utils returns `fps` as a per-video list (e.g. `[0.996]`), but the strict dataclass requires `int | float | None` — so a single-element list must be unwrapped to its scalar before it reaches the processor. On the Qwen3 path, `process_vision_info(..., return_video_metadata=True)` is used instead, which returns `(frames, metadata)` tuples per video (required by vLLM 0.27's Qwen3-VL parser, which sets `video_needs_metadata=True`) and never puts `fps` in `video_kwargs` at all — it travels inside each video's metadata instead — so the unwrap is a no-op on that path.
- **Omni audio via explicit content type** — vLLM V1's `use_audio_in_video` in `mm_processor_kwargs` is unreliable (audio embeddings silently dropped). Instead, add a standalone `{"type": "audio", "audio": ...}` content item alongside the video, then call `process_mm_info(messages, use_audio_in_video=False)`. This produces `<|audio_pad|>` tokens in the prompt so the model knows where to attend to audio. The audio item must point at a **pre-extracted WAV file**, not the video container — librosa/soundfile (used internally by `qwen_omni_utils.process_mm_info`) can't decode compressed containers like `.mp4` directly; older librosa fell back to `audioread` for that, but modern librosa dropped the fallback, so passing the video path raises `soundfile.LibsndfileError: Format not recognised`. Extract 16 kHz mono PCM WAV via ffmpeg first (see `_extract_audio_wav`) and pass that path instead.
- **vLLM shutdown noise** — when Modal tears down the container, vLLM's tensor-parallel workers log `KeyboardInterrupt` and "Engine core proc died unexpectedly". This is cosmetic — all work completes before shutdown. Caused by a race between Modal's container lifecycle and vLLM's multiprocess executor cleanup.
- **`enforce_eager=True`** — disables `torch.compile` and CUDA graph capture in vLLM, cutting cold-start time from ~4 min to ~1 min. The ~10-20% per-token generation slowdown is negligible for multimodal workloads where video/audio prefill dominates.
- WhisperX `large-v2` uses `float16` compute type; change to `int8` if GPU memory is tight.
- **flash-attn is deliberately not installed** — vLLM 0.27.1 bundles its own `vllm-flash-attn`, and no transformers attention path is used in `video-analyser.py`, so a separate flash-attn build is unnecessary.
- **Diarization raises, it doesn't silently degrade** — `audio-transcript.py` catches a gated/403 error from `whisperx.diarize.DiarizationPipeline` and raises a `RuntimeError` with accept-the-license instructions; it does not fall back to `SPEAKER_UNKNOWN` labels. whisperx 3.8.x loads `pyannote/speaker-diarization-community-1`, which bundles segmentation/embedding/PLDA internally, so accepting the older `speaker-diarization-3.1` conditions does nothing — that pipeline is never loaded.
- **`audioread` must stay an explicit dependency** — `qwen-omni-utils` imports `audioread` at module scope, but it's no longer pulled in transitively by librosa/soundfile. Removing it from the image breaks the import even though nothing in our code calls it directly. (The package is unpinned in the image; this isn't tied to a specific `qwen-omni-utils` version.)
- **`HF_HUB_ENABLE_HF_TRANSFER` is deprecated and inert** — `HF_XET_HIGH_PERFORMANCE=1` replaces it for fast HF Hub downloads. The old separate hf_transfer re-install step is gone, but the `hf_transfer` extra is still present in the image via `huggingface_hub[hf_transfer]` in the main `uv_pip_install` (`video-analyser.py:26`).
- **Base image CUDA version no longer determines the torch build** — `video-analyser.py`'s base image is CUDA 12.9, but nothing compiles from source anymore (flash-attn isn't installed; vLLM ships prebuilt wheels), so pip/uv resolves whatever torch build vLLM 0.27.1 pins — currently a **cu130** wheel — independent of the base image's CUDA toolkit. The infrastructure table's CUDA row describes the base image, not the installed torch wheel.
