# AI Video & Audio Analyser

> Serverless AI pipelines for deep video and audio understanding — running on GPU in the cloud with a single command.

This repo contains two Modal-powered tools:

| Script | What it does |
|---|---|
| [`video-analyser.py`](#video-analysis) | Analyze video files with Qwen VL / Qwen Omni — visual + audio understanding, scene descriptions, lecture notes, timestamps |
| [`audio-transcript.py`](#audio-transcription) | Transcribe audio **or video** with WhisperX — word-level timestamps, automatic language detection, speaker diarization |

Both scripts run entirely on Modal's serverless GPU infrastructure. You don't manage any servers — just `modal run` and go.

---

## Why Modal?

- **No GPU required locally** — all heavy lifting runs on A100/H100 instances spun up on demand
- **Pay per second** — containers scale to zero when idle
- **Model weights cached** — subsequent runs skip the download
- **Concurrent requests** — WhisperX supports up to 15 parallel transcriptions

---

## Setup

### 1. Install dependencies

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

Or set credentials manually:

```bash
export MODAL_TOKEN_ID="your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

Get your tokens at [modal.com/settings](https://modal.com/settings).

### 3. (For transcription with speaker diarization) Add a HuggingFace token

The diarization pipeline uses `pyannote/speaker-diarization`, which requires accepting the model license on Hugging Face and providing a token.

**Option A — Modal secret (recommended):**

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

**Option B — Environment variable:**

```bash
export HF_TOKEN=hf_your_token_here
```

---

## Video Analysis

Uses [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) multimodal models.

### Available models

| Key | Model | GPU | Capability |
|---|---|---|---|
| `qwen2-vl-2b` | Qwen2-VL 2B | A100 | Visual only, fast |
| `qwen2-vl-7b` | Qwen2-VL 7B | A100 | Visual only |
| `qwen3-vl-8b` | Qwen3-VL 8B | A100 | Visual only, strong _(default)_ |
| `qwen3-vl-235b` | Qwen3-VL 235B | 8× H100 | Visual only, SOTA |
| `qwen3-omni-30b-thinking` | Qwen3-Omni 30B | A100-80GB | Audio + Visual + reasoning |
| `qwen3-omni-30b-instruct` | Qwen3-Omni 30B | H100 | Audio + Visual |

### Quick start

```bash
# Default: Qwen3-VL 8B on a test video
modal run video-analyser.py

# Analyze your own video
modal run video-analyser.py \
  --video-url "https://example.com/lecture.mp4" \
  --prompt "Summarize the key points of this lecture as bullet-point notes."

# Use Qwen3-Omni for audio + visual understanding
modal run video-analyser.py \
  --model qwen3-omni-30b-thinking \
  --video-url "https://example.com/interview.mp4" \
  --prompt "Describe what is said and what is shown."

# Reduce memory with 8-bit quantization
modal run video-analyser.py \
  --model qwen3-omni-30b-thinking \
  --quantize-8bit

# Sample more frames (higher quality, slower)
modal run video-analyser.py --fps 2.0
```

### Output

Results are saved to `video_analysis_<model>.json`:

```json
{
  "analysis": "The video begins with...",
  "video_metadata": {
    "resolution": "1920x1080",
    "duration_seconds": 142.5,
    "fps": 25.0,
    "frames_sampled": 142
  },
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "chunked": false
}
```

### Long video support (chunking)

For videos longer than 2 minutes, `QwenOmniAnalyzer` automatically splits the video into 30-second chunks, processes each independently, and merges the results with preserved timestamps. This avoids GPU out-of-memory errors on long content.

### Web API

Deploy a persistent REST endpoint:

```bash
modal deploy video-analyser.py
```

```bash
# POST /analyze
curl -X POST "https://your-app.modal.run/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "prompt": "What happens in this video?",
    "model": "qwen3-vl-8b",
    "fps": 1.0
  }'

# GET /models
curl "https://your-app.modal.run/models"
```

---

## Audio Transcription

Uses [WhisperX](https://github.com/m-bain/whisperX) (`large-v2`) with CTranslate2 acceleration.

### Features

- **Word-level timestamps** — precise start/end time for every word
- **Automatic language detection** — or specify manually
- **Speaker diarization** — who said what, with `SPEAKER_00`, `SPEAKER_01` labels
- **Up to 15 concurrent jobs** — queue many files at once

### Quick start

WhisperX accepts **audio and video files** — ffmpeg extracts the audio track automatically from any video format (`.mp4`, `.mkv`, `.mov`, etc.).

```bash
# Transcribe audio with auto language detection + speaker diarization
modal run audio-transcript.py --audio-url "https://example.com/podcast.mp3"

# Transcribe the audio track from a video file
modal run audio-transcript.py --audio-url "https://example.com/lecture.mp4" --no-diarize

# Specify language, single speaker
modal run audio-transcript.py \
  --audio-url "https://example.com/talk.mp3" \
  --language en \
  --min-speakers 1 \
  --max-speakers 1

# Skip diarization (faster, no HF token needed)
modal run audio-transcript.py \
  --audio-url "https://example.com/audio.mp3" \
  --no-diarize
```

### Output files

| File | Contents |
|---|---|
| `transcript.json` | Raw WhisperX segments with timestamps |
| `transcript_align.json` | Word-level aligned segments |
| `transcript_dia.json` | Speaker-labeled segments (when diarization enabled) |

**Example `transcript_dia.json` entry:**

```json
{
  "start": 0.52,
  "end": 4.18,
  "text": " Welcome to the podcast.",
  "speaker": "SPEAKER_00",
  "words": [
    {"word": "Welcome", "start": 0.52, "end": 0.94, "score": 0.99, "speaker": "SPEAKER_00"},
    ...
  ]
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your machine                        │
│  modal run video-analyser.py  /  modal run audio-transcript.py  │
└───────────────────────┬─────────────────────────────────┘
                        │  Modal SDK (gRPC)
┌───────────────────────▼─────────────────────────────────┐
│                   Modal cloud                            │
│                                                          │
│  ┌─────────────────────────┐  ┌────────────────────────┐│
│  │   QwenVLAnalyzer        │  │  QwenOmniAnalyzer      ││
│  │   A100 · 64 GB VRAM     │  │  A100-80GB · 160 GB    ││
│  │   Qwen2/3-VL models     │  │  Qwen3-Omni models     ││
│  └─────────────────────────┘  └────────────────────────┘│
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │   WhisperXModel                                     ││
│  │   A100 · whisperx large-v2 · 15 concurrent inputs  ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │   Persistent Volume (model weight cache)            ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## Tips

- **First run is slow** — model weights are downloaded and cached. Subsequent runs are much faster.
- **`--fps 0.5`** — sample fewer frames to cut cost and latency on long videos.
- **`--max-pixels`** — lower this (e.g. `102400`) for faster processing at lower resolution.
- **Lecture notes prompt**: `"Transcribe step by step so the output can be used as lecture notes. Do not summarize."`
- **Diarization accuracy** — works best with clean audio and 2–4 distinct speakers.

---

## License

MIT
