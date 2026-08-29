# AI Video & Audio Analyser

> Serverless AI pipelines for deep video and audio understanding — running on GPU in the cloud with a single command.

This repo contains two Modal-powered tools:

| Script | What it does |
|---|---|
| [`video-analyser.py`](#video-analysis) | Analyze video files with Qwen VL / Qwen Omni — visual + audio understanding, scene descriptions, lecture notes, timestamps |
| [`audio-transcript.py`](#audio-transcription) | Transcribe audio **or video** with WhisperX — word-level timestamps, automatic language detection, speaker diarization |

Both scripts run entirely on Modal's serverless GPU infrastructure. You don't manage any servers — just `modal run` and go.

---

## Using uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies and the virtual environment.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv automatically)
uv sync

# Run any command inside the venv — no manual activation needed
uv run modal run video-analyser.py --video-url "..."
uv run modal deploy video-analyser.py

# Or activate the venv manually if you prefer
source .venv/bin/activate
modal run video-analyser.py --video-url "..."
```

> `uv run` is the recommended way — it ensures you're always using the correct venv without needing to activate/deactivate.

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
uv sync
```

This installs `modal`, `requests`, and `numpy` into a local `.venv` managed by [uv](https://docs.astral.sh/uv/).

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials (see comments in file)
```

### 3. Authenticate with Modal

```bash
modal setup   # interactive setup — OR fill MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in .env
```

Get your tokens at [modal.com/settings](https://modal.com/settings).

### 4. (Optional) HuggingFace token for speaker diarization

Speaker diarization (identifying *who* said what) is powered by [pyannote.audio](https://github.com/pyannote/pyannote-audio), a gated model on Hugging Face. WhisperX handles transcription on its own, but diarization requires this extra step. If you only need transcription, skip this — just pass `--no-diarize`.

**How to get the token:**

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Go to [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the license
3. Also accept the license at [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a `read` token is sufficient)

**Then provide it via one of:**

**Option A — Modal secret (recommended):**

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

**Option B — Environment variable / `.env` file:**

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
| `qwen3-omni-30b-thinking` | Qwen3-Omni 30B | 2× A100-80GB | Audio + Visual + reasoning |
| `qwen3-omni-30b-instruct` | Qwen3-Omni 30B | 2× H100 | Audio + Visual |

### Choosing the right model

The models fall into two families with fundamentally different capabilities:

**Qwen3-VL (Vision-Language)** — analyzes video frames and text only. No audio processing. Superior visual understanding (OCR, charts, fine details). The 235B variant is state-of-the-art for pure visual tasks but requires 8x H100 GPUs.

**Qwen3-Omni (Multimodal)** — processes audio and video together natively. Smaller and faster than VL-235B, but understands the relationship between what is said and what is shown. This is the right choice when your video has meaningful audio (speech, music, sound effects).

**Thinking vs Instruct** — The Omni family offers two variants:
- **Instruct** _(default)_ answers immediately without a reasoning step. Best for transcription, scene description, and general audio+video understanding. Faster and more token-efficient.
- **Thinking** uses chain-of-thought reasoning before answering. Better at complex tasks where audio and visual cues need to be connected (e.g. a lecturer pointing at a whiteboard while explaining a formula). Slower, uses more memory.

| Your use case | Recommended model |
|---|---|
| Quick visual description, no audio needed | `qwen3-vl-8b` _(default)_ |
| Highest accuracy for documents, diagrams, complex visuals | `qwen3-vl-235b` |
| Video with audio — transcription, description, general use | `qwen3-omni-30b-instruct` _(default for Omni)_ |
| Video with audio — complex reasoning, connecting cues | `qwen3-omni-30b-thinking` |
| Fast testing / prototyping | `qwen2-vl-2b` |

> **Tip:** If your video has important audio (speech, narration), always pick an Omni model. The VL models are "deaf" — they will analyze the visuals in detail but have no idea what is being said.

### Quick start

```bash
# Visual analysis with Qwen3-VL 8B (default model)
modal run video-analyser.py \
  --video-url "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4"

# Custom prompt
modal run video-analyser.py \
  --video-url "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4" \
  --prompt "Describe what happens in this video step by step."

# Use Qwen3-Omni for audio + visual understanding
modal run video-analyser.py \
  --model qwen3-omni-30b-instruct \
  --video-url "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4" \
  --prompt "Describe what is said and what is shown."

# Sample more frames (higher quality, slower)
modal run video-analyser.py --fps 2.0
```

### Output

Results are saved to `video_analysis_<model>.json` and printed to the terminal.

<details>
<summary><b>Example output</b> — Qwen3-Omni analyzing the <a href="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4">Bullrun sample video</a> (47s, audio + visual)</summary>

```
Model    : Qwen/Qwen3-Omni-30B-A3B-Thinking (QWEN3-OMNI)
Video    : WeAreGoingOnBullrun.mp4
Duration : 47.46s
Mode     : SINGLE PASS

0:00–0:05
A bald man with a beard and sunglasses drives a truck, looking toward the
passenger side. The scene cuts to a red pickup truck with "Black Magic,"
"Pepsi Max," and "FOX" logos parked on a cobblestone street, with the same
man sitting on the tailgate. The truck then drives down a highway. The driver
speaks: "Last year, the Smoking Tire won on the Bull Run Live rally in the
2010 Ford SVT Raptor."

0:05–0:10
The driver is shown inside the truck again, talking while driving through
traffic. The video then cuts to a wide shot of a large convoy of cars driving
on a city street, including a black Ferrari and a light purple car with
"Bullrun" branding. The driver continues: "I liked it so much, I bought one.
Here it is."

0:10–0:15
A blue Mitsubishi Lancer rally car with "Pepsi Max" and "Bullrun" logos drives
on a street. The scene cuts to a close-up of hands adjusting a radar detector
mounted inside the truck. The driver says: "We're going back to Bull Run this
year, of course, with help from our friends at Black Magic. And we're so
serious about it, we got two Valentine One radar detectors."

0:15–0:20
A close-up shows the radar detector with its red lights flashing as it's
adjusted. The driver looks out the window at a traffic light and says,
"Oh yeah." The scene then cuts to the truck driving through an intersection.

0:20–0:25
A black Chevrolet Corvette drives on a highway. The camera then shows a red
car with a white "Bullrun" logo driving on a highway, followed by a black
Ferrari driving alongside it. The driver states: "So we're all set up. And
the reason we got two is because we're going to be going a little bit faster."

0:25–0:30
A red Ford Mustang Shelby GT500 with white racing stripes is shown in a
studio setting. The scene then cuts to a highway where people are standing
on the roof of a white convertible car. The driver continues: "We got a 2011
Shelby GT500 with a 550 horsepower all-aluminum V8."

0:30–0:35
Multiple rally cars, including a silver Audi and a white Ferrari with
"Bullrun" branding, drive on a highway. The camera then shows a black pickup
truck with "Pepsi Max" and "Bullrun" logos driving on the highway, with a
woman in the bed of the truck. The driver says: "We are going to be right in
the action, bringing you guys a video every single day, live from the Bull Run
rally, July 9th to 16th."

0:35–0:40
The driver is shown again inside the truck, talking while driving through city
traffic. The text "BlackMagicShine.com" appears at the bottom of the screen.
He continues: "And the only place to watch it is on BlackMagicShine.com or
right here on the Smoking Tire."

0:40–0:47
The camera shows the dashboard and the road ahead as the truck drives through
an intersection. The driver finishes: "We're going back to Bull Run this year,
of course, with help from our friends at Black Magic." A close-up shows the
radar detector with its red lights flashing. The driver says, "Oh yeah." The
video ends with the truck driving on the road.
```

</details>

The JSON output file includes full metadata:

```json
{
  "analysis": "0:00–0:05\nA bald man with a beard...",
  "video_metadata": {
    "resolution": "1280x720",
    "duration_seconds": 47.46,
    "fps": 24.0,
    "frames_sampled": 46,
    "audio_enabled": true,
    "audio_extracted": true
  },
  "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
  "chunked": false
}
```

### Long video support (chunking)

For videos longer than 2 minutes, `QwenOmniAnalyzer` automatically splits the video into 30-second chunks using `ffmpeg -c copy` (stream copy, no re-encoding), submits all chunks in a single `llm.generate()` call for concurrent processing via vLLM's continuous batching, and merges the results with preserved timestamps. This avoids GPU out-of-memory errors since multimodal models must hold video frames and audio in VRAM simultaneously.

Chunking only applies to `video-analyser.py`. The audio transcription pipeline (`audio-transcript.py`) does **not** need chunking — WhisperX processes audio in internal batches and handles long files natively without running out of memory.

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
    "video_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4",
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
│  │   A100 · vLLM (tp=1)   │  │  2×A100-80GB · vLLM    ││
│  │   Qwen2/3-VL models     │  │  Qwen3-Omni (tp=2)    ││
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
