# Dependency Upgrade (Phases 1 & 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring both Modal scripts from their Feb/Mar 2026 pins onto the August 2026 stack — Modal client 1.5.x, whisperx 3.8.6, vLLM 0.27.1 on torch 2.13 / CUDA 12.9 — without changing any model or feature behaviour.

**Architecture:** Two independent Modal images that must stay independent: `audio-transcript.py` is locked to the torch 2.8 line by whisperx, while `video-analyser.py` moves to the torch 2.13 line by vLLM. The upgrade is sequenced so exactly one variable changes per task, each with its own Modal verification run, so a failure points at a single cause. Verification is a real `modal run` — there is no local GPU capable of running these models — with a cheap CPU/L4-class version assertion (`verify_env`) added as the fast gate before any expensive GPU run.

**Tech Stack:** Modal 1.5.x, vLLM 0.27.1, PyTorch 2.13.0, transformers 5.5.3+, whisperx 3.8.6, uv, CUDA 12.9 devel base image, Python 3.12 in-image.

**Spec:** This plan; source research is in the session transcript of 2026-08-29 (model/infra landscape review). Phases 3 (Qwen3.5 model refresh) and 4 (GPU memory snapshots, single-H200 Omni) are explicitly **out of scope**.

## Global Constraints

Exact values, verified against PyPI metadata on 2026-08-29. Do not substitute.

- **vLLM 0.27.1 hard pins:** `torch==2.13.0`, `torchvision==0.28.0`, `torchaudio==2.11.0`, `transformers>=5.5.3`. Requires-python `>=3.10,<3.15`.
- **whisperx 3.8.6 hard pins:** `torch~=2.8.0`, `torchaudio~=2.8.0`, `torchvision~=0.23.0`, `torchcodec>=0.6.0,<0.8.0`, `pyannote-audio>=4.0.0`, `faster-whisper>=1.2.0`, `ctranslate2>=4.5.0`, `transformers>=4.48.0`. Requires-python `>=3.10,<3.14`.
- **Never install torch explicitly in the video-analyser image.** vLLM pins it exactly; let `uv_pip_install("vllm==0.27.1")` resolve it. This mirrors Modal's own vLLM example, which installs nothing but vLLM.
- **Base image for video-analyser:** `nvidia/cuda:12.9.0-devel-ubuntu22.04` with `add_python="3.12"` (same as Modal's current vLLM example).
- **The two images stay separate.** torch 2.13 (vLLM) and torch 2.8 (whisperx) cannot coexist. Do not attempt to share an image.
- **Modal client:** `modal>=1.5,<2` in `pyproject.toml`. Modal 1.6.0 will flip Sandbox V2 on by default and remove already-deprecated APIs; this repo uses none of them, but pin the major version.
- **No behaviour changes.** No new models, no GPU changes, no `enforce_eager` changes, no snapshot APIs. Pins and dead code only.
- **Every verification runs on Modal.** The local machine has an 8 GB laptop GPU and cannot run any of these models.
- **Approximate GPU cost per verification run** is noted on each task so the executor can batch expensive runs. Use the sample URL `https://media.w3.org/2010/05/sintel/trailer.mp4` for every smoke test — verified 2026-08-29: HTTP 200, 52.2s, h264 + aac stereo, 4.4 MB. It is public, short, under the 120s chunking threshold, and has an audio track, so it doubles as the transcription fixture. The README's old commondatastorage.googleapis.com sample now returns 403 and must not be used.

---

## File Structure

| File | Responsibility | Changes in this plan |
|-|-|-|
| `pyproject.toml` | Local CLI env only (the `modal` client, dotenv, requests, numpy) | Pin `modal>=1.5,<2` |
| `uv.lock` | Lockfile for the local env | Regenerated |
| `.gitignore` | Keep media/scratch artifacts out of git | Add downloaded-media and log patterns |
| `audio-transcript.py` | WhisperX image + `WhisperXModel` class | Image block only (lines 5–24) |
| `video-analyser.py` | vLLM image + `QwenVLAnalyzer` + `QwenOmniAnalyzer` + FastAPI endpoint | Image block (lines 12–64); new `verify_env` function |
| `CLAUDE.md` | Developer context, infra table, gotchas | Version table + flash-attn/vLLM gotchas |
| `README.md` | User-facing setup | Version mentions only |

---

### Task 1: Repo hygiene and Modal client bump

Untracked media and log files currently pollute `git status`, which makes the per-task commits in this plan noisy and error-prone. Fix that first, then bump the client.

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated, do not hand-edit)

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `git status`, and a local `modal` CLI at 1.5.x that later tasks invoke as `modal run` / `modal serve`.

- [ ] **Step 1: Confirm the current client version (this is the "before" measurement)**

```bash
.venv/bin/python -c "import modal; print(modal.__version__)"
```

Expected: `1.3.3`

- [ ] **Step 2: Add ignore patterns for the untracked junk**

Append to `.gitignore`:

```gitignore
# Downloaded media and scratch artifacts
download
wget-log*
*.aac
*.mp3
*.mp4
*.wav
transcript*.json
video_analysis_*.json
backup*.json
IQ*nav=*
```

- [ ] **Step 3: Verify the working tree is clean apart from intended edits**

```bash
git status --porcelain
```

Expected: only ` M README.md`, ` M video-analyser.py`, ` M .gitignore`, ` M pyproject.toml` and the new `docs/` path. No `?? download`, no `?? wget-log`, no `?? IQCEZ...` entries. If any junk still shows, add its pattern and re-run.

- [ ] **Step 4: Commit the pre-existing working-tree changes separately**

The repo already carried uncommitted Omni audio-extraction fixes before this plan started. They are unrelated to the upgrade and must not be mixed into an upgrade commit.

```bash
git add README.md video-analyser.py
git commit -m "fix: extract chunk audio to wav for Omni, preserve source filename"
```

- [ ] **Step 5: Pin the Modal client**

In `pyproject.toml`, change the `dependencies` list entry `"modal",` to:

```toml
    "modal>=1.5,<2",
```

- [ ] **Step 6: Resolve and install**

```bash
uv lock --upgrade-package modal && uv sync
```

- [ ] **Step 7: Verify the new client version and that it can reach Modal**

```bash
.venv/bin/python -c "import modal; print(modal.__version__)"
.venv/bin/modal app list
```

Expected: version `1.5.x` (1.5.4 or newer), and `modal app list` returns a table without an auth or version error. If `modal app list` fails on credentials, the `.env` in this repo holds `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` — export them or run `modal setup`.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml uv.lock docs/superpowers/plans/2026-08-29-dependency-upgrade.md
git commit -m "chore: pin modal>=1.5,<2, ignore scratch media, add upgrade plan"
```

---

### Task 2: whisperx 3.8.1 → 3.8.6

whisperx 3.8.6 keeps the torch 2.8 line, so the existing `torch==2.8.0` / `torchaudio==2.8.0` pins stay correct. It adds a `torchvision~=0.23.0` requirement that is currently unpinned in the image, and `torchcodec==0.7.0` remains inside the new `>=0.6.0,<0.8.0` range. Pin torchvision explicitly for the same reason the other two are pinned: uv resolves each `uv_pip_install` step independently.

**Files:**
- Modify: `audio-transcript.py:5-24` (the `image` block)

**Interfaces:**
- Consumes: Modal CLI from Task 1.
- Produces: a working `whisperx-transcriber` app on whisperx 3.8.6. No Python API changes — `whisperx.load_model`, `load_align_model`, `align`, `diarize.DiarizationPipeline`, `assign_word_speakers` all keep their current signatures.

- [ ] **Step 1: Establish the baseline — run the current app once and keep the output**

```bash
.venv/bin/modal run audio-transcript.py --audio-url "https://media.w3.org/2010/05/sintel/trailer.mp4" --no-diarize
```

Expected: completes and writes transcript JSON. Save the printed transcript text somewhere outside the repo (e.g. the scratchpad) — Step 5 compares against it. Approximate cost: A100, ~2–4 min.

- [ ] **Step 2: Bump the pins**

In `audio-transcript.py`, replace the second `uv_pip_install` block:

```python
    .uv_pip_install(
        "whisperx==3.8.6",
        "ffmpeg-python",
        # Pin torch + torchaudio + torchvision here too — uv resolves each step
        # independently, and whisperx 3.8.6 requires torch~=2.8.0, torchaudio~=2.8.0,
        # torchvision~=0.23.0 and torchcodec>=0.6.0,<0.8.0 (pyannote-audio 4.x).
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchvision==0.23.0",
        "torchcodec==0.7.0",
        uv_version="0.10.3",
    )
```

- [ ] **Step 3: Pre-check the resolution locally (cheap failure gate)**

There is no `modal build` command — images build at the start of `modal run`. Catch a bad pin before involving Modal at all by resolving the same set with uv locally:

```bash
printf '%s\n' 'whisperx==3.8.6' 'ffmpeg-python' 'torch==2.8.0' 'torchaudio==2.8.0' \
  'torchvision==0.23.0' 'torchcodec==0.7.0' \
  > /tmp/whisperx-req.txt
uv pip compile /tmp/whisperx-req.txt --python-version 3.12 -o /dev/null
```

Expected: resolves without error. A conflict here means a pin above is wrong — uv's error names the conflicting requirement. (A `modal run` would surface the same error, also before any GPU is provisioned, but this is faster.)

- [ ] **Step 4: Run the same transcription on the new image**

```bash
.venv/bin/modal run audio-transcript.py --audio-url "https://media.w3.org/2010/05/sintel/trailer.mp4" --no-diarize
```

Expected: completes, transcript is substantively the same as Step 1 (word timings may differ slightly; the words should not).

- [ ] **Step 5: Verify diarization specifically**

This is the risky half of the whisperx bump. 3.8.x switched to `pyannote/speaker-diarization-community-1`, and when the HF token has not accepted **that** model's conditions, diarization does not error — it silently returns `SPEAKER_UNKNOWN` labels. Accepting the older `speaker-diarization-3.1` conditions does nothing.

```bash
.venv/bin/modal run audio-transcript.py --audio-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: segments carry real labels (`SPEAKER_00`, `SPEAKER_01`, …).
Failure mode to watch for: every segment labelled `SPEAKER_UNKNOWN`, with no error raised. If that happens, visit <https://huggingface.co/pyannote/speaker-diarization-community-1>, accept the conditions with the account owning `HF_TOKEN`, and re-run. Do not "fix" this in code.

- [ ] **Step 6: Commit**

```bash
git add audio-transcript.py
git commit -m "chore: upgrade whisperx to 3.8.6, pin torchvision"
```

---

### Task 3: Drop the flash-attn build from the video-analyser image

`flash-attn==2.8.3` is compiled from source on every image rebuild and is not used by anything: the code path is 100% vLLM, which bundles its own `vllm-flash-attn`, and `grep` finds no `attn_implementation` or `flash_attn` reference in either script. `TORCH_CUDA_ARCH_LIST` existed only to steer that compile. Removing both here — *before* the vLLM bump — keeps the two changes independently revertible.

**Files:**
- Modify: `video-analyser.py:12-64` (the `image` block)

**Interfaces:**
- Consumes: Modal CLI from Task 1.
- Produces: an image with no source compilation step. `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_HOME` must survive unchanged — they are load-bearing.

- [ ] **Step 1: Confirm nothing references flash-attn**

```bash
grep -rn "flash_attn\|flash_attention\|attn_implementation\|TORCH_CUDA_ARCH_LIST" video-analyser.py audio-transcript.py
```

Expected: no matches other than the image-block lines you are about to delete. If application code *does* reference it, stop and re-plan — the assumption behind this task is wrong.

- [ ] **Step 2: Remove the build-deps + flash-attn steps**

Delete these two steps from the image chain:

```python
    # Build deps + compile flash-attn AFTER vLLM so it links against the final PyTorch ABI
    .uv_pip_install("packaging", "wheel", "setuptools", "ninja", "psutil", uv_version="0.10.3")
    .run_commands(
        "pip install flash-attn==2.8.3 --no-build-isolation"
    )
```

- [ ] **Step 3: Remove the now-dead arch list from `.env()`**

Delete only this line from the `.env({...})` dict:

```python
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
```

Leave `HF_HUB_ENABLE_HF_TRANSFER`, `HF_HOME` and `VLLM_WORKER_MULTIPROC_METHOD` exactly as they are.

- [ ] **Step 4: Rebuild the image and time it**

The image rebuilds on the next `modal run`; Step 5 triggers it. Time that run and compare against Task 3's baseline — image build is the first phase of its output.

```bash
time .venv/bin/modal run video-analyser.py --model qwen2-vl-2b --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: the build phase is noticeably shorter than before — the flash-attn source compile was its dominant cost. This is the same command as Step 5; run it once and use it for both.

- [ ] **Step 5: Prove the VL path still works on the old vLLM**

```bash
.venv/bin/modal run video-analyser.py --model qwen2-vl-2b --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: a coherent description of the video. This isolates "removing flash-attn broke nothing" from the vLLM bump that follows. Approximate cost: A100, ~3–5 min.

- [ ] **Step 6: Commit**

```bash
git add video-analyser.py
git commit -m "chore: drop unused flash-attn build and TORCH_CUDA_ARCH_LIST"
```

---

### Task 4: Add `verify_env` — the failing test for the vLLM bump

Before touching the pins, add a cheap Modal function that asserts the target versions. It fails on the current image (vLLM 0.13.0, transformers 4.x) and passes once Task 5 lands. It runs on an L4, so it costs cents and catches resolution mistakes before any A100 starts.

**Files:**
- Modify: `video-analyser.py` — add a new function immediately after the `MODELS` dict and before `hf_secret` (around line 133)

**Interfaces:**
- Consumes: the module-level `image` and `app` objects.
- Produces: `verify_env()` — a `@app.function(gpu="L4", timeout=600)` taking no arguments, returning `None`, raising `AssertionError` on a version mismatch. Task 5 uses it as its pass/fail gate; keep it in the repo afterwards as a regression check for future bumps.

- [ ] **Step 1: Write the failing test**

Insert into `video-analyser.py`:

```python
@app.function(gpu="L4", timeout=600)
def verify_env():
    """Assert the image resolved to the intended inference stack.

    Cheap gate before any expensive GPU smoke test — a resolution mistake in the
    image definition surfaces here in a couple of minutes instead of on an A100.
    """
    import torch
    import transformers
    import vllm
    from packaging.version import Version

    versions = {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }
    for name, value in versions.items():
        print(f"{name}: {value}")

    assert Version(vllm.__version__) >= Version("0.27.1"), versions
    assert Version(torch.__version__.split("+")[0]) >= Version("2.13.0"), versions
    assert Version(transformers.__version__) >= Version("5.5.3"), versions
    assert torch.cuda.is_available(), "CUDA not available in container"

    # The multimodal helpers are imported inside the analyzer methods; confirm
    # here that they still import against the resolved transformers version.
    from qwen_omni_utils import process_mm_info  # noqa: F401
    from qwen_vl_utils import process_vision_info  # noqa: F401

    print("environment OK")
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
.venv/bin/modal run video-analyser.py::verify_env
```

Expected: `AssertionError` naming the current versions — vllm `0.13.0`, torch `2.7.0+cu128`, transformers `4.x`. A *pass* here means the image is not what this plan assumes; stop and investigate. Approximate cost: L4, ~1–2 min.

- [ ] **Step 3: Commit the failing test**

```bash
git add video-analyser.py
git commit -m "test: add verify_env version gate for the inference image"
```

---

### Task 5: vLLM 0.27.1 + torch 2.13 + CUDA 12.9

The core of Phase 2. Three coupled changes that cannot be split: the base image, removal of the manual torch install, and the vLLM/transformers pins.

**Files:**
- Modify: `video-analyser.py:7-64` (CUDA version constant and the whole `image` block)

**Interfaces:**
- Consumes: `verify_env()` from Task 4 as the gate.
- Produces: an image where `import vllm` yields 0.27.1 on torch 2.13.0. Tasks 6 and 7 exercise it end to end.

- [ ] **Step 1: Bump the base image**

Change the CUDA constant at the top of `video-analyser.py`:

```python
cuda_version = "12.9.0"
```

Leave `flavor = "devel"` and `operating_sys = "ubuntu22.04"` alone. (`devel` is now heavier than strictly needed since nothing compiles, but it matches Modal's published vLLM example and rules out a class of missing-toolchain surprises. Slimming it is a separate, optional change.)

- [ ] **Step 2: Delete the manual PyTorch install step**

Remove this entire step from the image chain — vLLM 0.27.1 pins `torch==2.13.0`, `torchvision==0.28.0` and `torchaudio==2.11.0` itself, and a pre-installed cu128 torch only creates a conflict for uv to unwind:

```python
    # Install PyTorch with CUDA 12.8
    .uv_pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        index_url="https://download.pytorch.org/whl/cu128",
        uv_version="0.10.3",
    )
```

- [ ] **Step 3: Raise the transformers floor in the support-package step**

vLLM 0.27.1 requires `transformers>=5.5.3` — a major-version jump from the current `>=4.48.0`. Stating the real floor in the earlier step stops uv from installing a 4.x and then churning it. In the second `uv_pip_install` block, change:

```python
        "transformers>=5.5.3",
```

Leave every other package in that block unchanged (`fastapi[standard]`, `uvicorn[standard]`, `pydantic>=2.0`, `accelerate>=0.26.0`, `bitsandbytes>=0.44.0`, `qwen-vl-utils>=0.0.14`, `qwen-omni-utils`, `huggingface_hub[hf_transfer]`, `safetensors`, `tokenizers`, `einops`, `Pillow`, `numpy`, `opencv-python-headless`, `av`, `decord`, `requests`, `tqdm`, `pyyaml`).

- [ ] **Step 4: Bump vLLM**

```python
    # Install vLLM last so its exact torch/transformers pins win
    .uv_pip_install("vllm==0.27.1", uv_version="0.10.3")
```

- [ ] **Step 5: Pre-check the resolution locally before involving Modal**

```bash
printf '%s\n' 'vllm==0.27.1' 'transformers>=5.5.3' 'accelerate>=0.26.0' \
  'bitsandbytes>=0.44.0' 'qwen-vl-utils>=0.0.14' 'qwen-omni-utils' \
  'huggingface_hub[hf_transfer]' 'av' 'decord' 'opencv-python-headless' \
  'fastapi[standard]' 'pydantic>=2.0' \
  > /tmp/vllm-req.txt
uv pip compile /tmp/vllm-req.txt --python-version 3.12 -o /dev/null
```

Expected: resolves, pulling `torch==2.13.0`, `torchvision==0.28.0`, `torchaudio==2.11.0`. If uv reports a conflict, the offending requirement is named in the error — the most likely culprit is a package that caps `transformers<5`. Resolve by raising that package's version, never by lowering the vLLM pin. Note this checks resolvability only; the real image build happens on the next `modal run` in Step 6.

- [ ] **Step 6: Run the test from Task 4 and watch it pass**

```bash
.venv/bin/modal run video-analyser.py::verify_env
```

Expected: prints vllm `0.27.1`, torch `2.13.0`, transformers `5.5.x`, a CUDA version, then `environment OK`.
If the `qwen_vl_utils` / `qwen_omni_utils` imports fail here, that is the transformers 5.x incompatibility this plan is most worried about — do not paper over it; record the traceback and handle it in Task 6, which has the fallback strategy.

- [ ] **Step 7: Commit**

```bash
git add video-analyser.py
git commit -m "chore: upgrade to vLLM 0.27.1 on torch 2.13 / CUDA 12.9"
```

---

### Task 6: VL path end-to-end on the new stack

**Files:**
- Modify (only if the smoke test fails): `video-analyser.py:150-320` (`QwenVLAnalyzer.setup` and `analyze_video`)

**Interfaces:**
- Consumes: the image from Task 5.
- Produces: a verified `QwenVLAnalyzer`. `analyze_video()` keeps its current signature and its return dict keys — no caller-visible change.

- [ ] **Step 1: Smallest model first**

```bash
.venv/bin/modal run video-analyser.py --model qwen2-vl-2b --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: a coherent description. Approximate cost: A100, ~3–5 min (first run also re-verifies the cached weights in the `qwen3-vl-cache` volume).

- [ ] **Step 2: If it fails, work through these three known-risk points in order**

Each is a real API surface that moved between vLLM 0.13 and 0.27. Apply only the one the traceback points at.

**(a) `LLM(...)` rejects a keyword** in `QwenVLAnalyzer.setup` (`video-analyser.py:170-183`). The current call passes `gpu_memory_utilization`, `tensor_parallel_size`, `limit_mm_per_prompt`, `max_num_seqs`, `max_model_len`, `dtype`, `mm_encoder_tp_mode`, `enable_expert_parallel`, `enforce_eager`. If one is rejected, check its replacement in vLLM's engine-args docs and rename it. Do **not** drop `enforce_eager` — cold-start behaviour is deliberately out of scope for this plan.

**(b) `process_vision_info` fails against transformers 5.x.** `qwen-vl-utils` last released 0.0.14 in Sept 2025. If it breaks, the fallback is to let vLLM's own processor do the work: pass the chat-template prompt plus `{"video": <path>}` as `multi_modal_data` and drop the `mm_processor_kwargs` that `return_video_kwargs=True` supplies. Verify against vLLM's multimodal-inputs docs for 0.27 before writing it.

**(c) `mm_processor_kwargs` shape changed.** `video_kwargs` from `process_vision_info(..., return_video_kwargs=True)` is currently forwarded verbatim. If vLLM rejects a key, drop that key rather than the whole dict.

- [ ] **Step 3: Re-run until Step 1's command produces a description**

```bash
.venv/bin/modal run video-analyser.py --model qwen2-vl-2b --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

- [ ] **Step 4: Verify the Qwen3-VL path too — it is a different vLLM architecture class than Qwen2-VL**

```bash
.venv/bin/modal run video-analyser.py --model qwen3-vl-8b --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: a coherent description. Approximate cost: A100, ~5 min. Skip `qwen3-vl-235b` — it needs 8×H100 and proves nothing the 8B does not.

- [ ] **Step 5: Verify the web endpoint still starts**

```bash
.venv/bin/modal serve video-analyser.py
```

Expected: prints an endpoint URL and stays up. `Ctrl-C` after the URL appears — `@modal.asgi_app()` now requires the decorated function to be nullary, and `fastapi_app()` already is, so this is a confirmation, not a fix.

- [ ] **Step 6: Commit**

```bash
git add video-analyser.py
git commit -m "fix: adapt VL analyzer to vLLM 0.27 API"
```

If Steps 2–4 required no code change, skip the commit and note "no changes needed" in the task record.

---

### Task 7: Omni path end-to-end on the new stack

The Omni analyzer is the more fragile of the two: tensor parallelism across 2 GPUs, chunked video, and audio passed as an explicit modality.

**Files:**
- Modify (only if the smoke test fails): `video-analyser.py:321-700` (`QwenOmniAnalyzer`)

**Interfaces:**
- Consumes: the image from Task 5, verified by Task 6.
- Produces: a verified `QwenOmniAnalyzer`. `analyze_video()` signature and return dict unchanged.

- [ ] **Step 1: Run the short path (no chunking)**

The sample video is under the 120 s `min_duration_for_chunking` threshold, so this exercises the single-segment path first.

```bash
.venv/bin/modal run video-analyser.py --model qwen3-omni-30b-instruct --video-url "https://media.w3.org/2010/05/sintel/trailer.mp4"
```

Expected: a description that references **both** what is seen and what is heard. Approximate cost: 2×A100-80GB, ~8–12 min. If the output only ever describes visuals, the audio modality is being dropped — see Step 2(c).

- [ ] **Step 2: If it fails, work through these known-risk points**

**(a) Tensor-parallel workers die at startup.** `VLLM_WORKER_MULTIPROC_METHOD=spawn` must still be set in the image `.env()` — confirm Task 3 did not remove it. vLLM V1's executor forks by default and pynvml's `cuInit()` before fork breaks CUDA in the children.

**(b) `process_mm_info` fails against transformers 5.x.** `qwen-omni-utils` is at 0.0.9 (Feb 2026). Same fallback shape as Task 6 Step 2(b): hand vLLM the paths directly as `multi_modal_data` and let its processor build the embeddings.

**(c) Audio silently missing from the output.** The working design is deliberate and must be preserved: audio is passed as a **separate** `{"type": "audio", "audio": <wav path>}` content item alongside the video, with `process_mm_info(messages, use_audio_in_video=False)`. vLLM's `use_audio_in_video` processor kwarg drops audio embeddings silently. The uncommitted fix from Task 1 Step 4 also extracts each chunk's audio to 16 kHz mono WAV with ffmpeg because chunk files may carry no audio stream and librosa cannot decode `.mp4` containers. Do not "simplify" either of these.

- [ ] **Step 3: Verify the chunked path on a video longer than 120 s**

```bash
.venv/bin/modal run video-analyser.py --model qwen3-omni-30b-instruct --video-url "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"
```

Expected: log lines showing multiple chunks submitted in one `llm.generate()` call, and per-chunk output with timeline-relative references. Approximate cost: 2×A100-80GB, ~15 min. If that URL 404s, substitute any public MP4 over two minutes long and note which you used.

- [ ] **Step 4: Commit**

```bash
git add video-analyser.py
git commit -m "fix: adapt Omni analyzer to vLLM 0.27 API"
```

Skip if no code change was needed.

---

### Task 8: Update the documentation to match reality

**Files:**
- Modify: `CLAUDE.md` (infrastructure table, "Common gotchas")
- Modify: `README.md` (any version mentions surfaced by grep)

**Interfaces:**
- Consumes: the verified state from Tasks 2–7.
- Produces: docs an agent can trust on the next session. `CLAUDE.md` is loaded into context automatically, so a stale version table there actively misleads.

- [ ] **Step 1: Update the infrastructure table in `CLAUDE.md`**

Set the `video-analyser` column to: base image `nvidia/cuda:12.9.0-devel-ubuntu22.04`, CUDA `12.9`, PyTorch `2.13.0 (via vLLM)`, vLLM `0.27.1`, and **delete the `flash-attn` row entirely**. Set the `audio-transcript` column's `whisperx` value to `3.8.6`.

- [ ] **Step 2: Fix the gotchas that this upgrade invalidated**

In "Common gotchas": delete nothing about `VLLM_WORKER_MULTIPROC_METHOD` (still true), `enforce_eager` (still true and still in the code), `max_num_seqs=8` (unchanged), the multi-modal data dict, `mm_processor_kwargs`, the Omni explicit-audio note, `ffmpeg -c copy` cut points, or vLLM shutdown noise. Add one line recording that flash-attn is deliberately not installed because vLLM bundles `vllm-flash-attn` and no transformers attention path is used. If Tasks 6 or 7 changed an API call, replace the matching gotcha with what is now true.

- [ ] **Step 3: Add a diarization gotcha to `CLAUDE.md`**

```markdown
- **Diarization silently degrades** — whisperx 3.8.x uses `pyannote/speaker-diarization-community-1`. If the HF account behind `HF_TOKEN` has not accepted *that* model's conditions, diarization returns `SPEAKER_UNKNOWN` for every segment instead of raising. Accepting `speaker-diarization-3.1` does nothing.
```

- [ ] **Step 4: Sweep the README for stale versions**

```bash
grep -n "2\.7\.0\|0\.13\.0\|12\.8\|flash-attn\|3\.8\.1\|large-v2" README.md
```

Update whatever it finds. `large-v2` is still what `audio-transcript.py` loads, so leave that alone unless the code changed.

- [ ] **Step 5: Verify the claims in the docs against the code one last time**

```bash
grep -n "cuda_version\|vllm==\|whisperx==\|transformers>=" video-analyser.py audio-transcript.py
```

Every version in `CLAUDE.md` must match this output exactly.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update stack versions and gotchas for the 2026-08 upgrade"
```

---

## Rollback

Every task is a single commit on `chore/dep-upgrade-2026-08`, so `git revert <sha>` undoes one change without disturbing its neighbours. Modal images are content-addressed: reverting the image definition restores the previously built image from cache rather than rebuilding it, so a rollback is fast. `main` is untouched throughout.

## Out of scope — do not do these here

- Adding Qwen3.5 models (9B / 27B / 122B-A10B) — Phase 3, separate plan.
- GPU memory snapshots, dropping `enforce_eager`, single-H200 Omni — Phase 4.
- Removing `bitsandbytes` / `decord` (unused-looking, but unverified) — separate cleanup.
- Migrating `@modal.asgi_app()` to `@app.server()`.
- Any change to `audio-transcript.py` beyond the version pins in Task 2.
