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
| vLLM | 0.13.0 | — |
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

Two Modal classes dispatched based on model series, both using **vLLM** for inference:

- **`QwenVLAnalyzer`** — visual-only, vLLM `LLM()` with `process_vision_info` for multimodal data
- **`QwenOmniAnalyzer`** — audio+visual, vLLM `LLM()` with tensor parallelism (tp=2) on 2x A100-80GB

Routing logic lives in `main()` and `fastapi_app()`: if `model_config["series"] == "qwen3-omni"` → `QwenOmniAnalyzer`, otherwise → `QwenVLAnalyzer`.

**Tensor parallelism**: each model in `MODELS` has a `tensor_parallel_size` field. Omni models use tp=2 (2 GPUs), VL models use tp=1 (single GPU), and the 235B model uses tp=8.

**Batched chunk processing** (`QwenOmniAnalyzer`): videos longer than `min_duration_for_chunking` (default 120s) are split into segments via `ffmpeg -c copy`, then all chunks are submitted in a single `llm.generate()` call — vLLM's continuous batching processes them concurrently across GPUs.

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
    "tensor_parallel_size": 1,  # number of GPUs for vLLM tensor parallelism
    "max_model_len": 32768,
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
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** is required globally — vLLM V1's multiprocess executor forks by default, but pynvml calls `cuInit()` before fork which breaks CUDA in child workers. Spawn creates fresh processes and avoids this. (V0 engine was removed in vLLM 0.13.0; `VLLM_USE_V1=0` is no longer recognized.)
- **`max_num_seqs=8`** limits concurrent batch processing in vLLM. For chunked videos, all chunks are submitted in one `llm.generate()` call but vLLM queues beyond 8.
- **Multi-modal data dict** — vLLM expects `{"prompt": ..., "multi_modal_data": {"video": ..., "audio": ...}}` format. The processor's `process_vision_info` / `process_mm_info` output feeds directly into `multi_modal_data`.
- **`mm_processor_kwargs`** — required in the vLLM input dict for VL models. `process_vision_info(..., return_video_kwargs=True)` returns a third value (`video_kwargs`) that must be passed through.
- **Omni audio via explicit content type** — vLLM V1's `use_audio_in_video` in `mm_processor_kwargs` is unreliable (audio embeddings silently dropped). Instead, add `{"type": "audio", "audio": video_path}` as an explicit content item in the message alongside the video. This produces `<|audio_pad|>` tokens in the prompt so the model knows where to attend to audio. Call `process_mm_info(messages, use_audio_in_video=False)` since audio is handled as a standalone modality.
- **vLLM shutdown noise** — when Modal tears down the container, vLLM's tensor-parallel workers log `KeyboardInterrupt` and "Engine core proc died unexpectedly". This is cosmetic — all work completes before shutdown. Caused by a race between Modal's container lifecycle and vLLM's multiprocess executor cleanup.
- **`enforce_eager=True`** — disables `torch.compile` and CUDA graph capture in vLLM, cutting cold-start time from ~4 min to ~1 min. The ~10-20% per-token generation slowdown is negligible for multimodal workloads where video/audio prefill dominates.
- WhisperX `large-v2` uses `float16` compute type; change to `int8` if GPU memory is tight.
