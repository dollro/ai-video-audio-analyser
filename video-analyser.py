import modal
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Configure CUDA environment
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libcudnn8",
        "libcudnn8-dev",
    )
    # Install PyTorch first
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install flash-attn from official pre-built wheel
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .pip_install(
        "packaging",
        "wheel",
        "setuptools",
        "ninja",
        "fastapi[standard]",
        "uvicorn[standard]",
        "pydantic>=2.0",
        "transformers>=4.48.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.44.0",
        "qwen-vl-utils>=0.0.8",
        "qwen-omni-utils",
        "huggingface_hub[hf_transfer]",
        "safetensors",
        "tokenizers",
        "einops",
        "Pillow",
        "numpy<2.0",
        "opencv-python-headless",
        "av",
        "decord",
        "requests",
        "tqdm",
        "psutil",
        "pyyaml",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
    })
)

app = modal.App("qwen-video-analyzer", image=image)

CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("qwen3-vl-cache", create_if_missing=True)

# Available model configurations
MODELS = {
    # Qwen2-VL models (for comparison/fallback)
    "qwen2-vl-2b": {
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "gpu": "A100",
        "memory": 32768,
        "description": "Smallest Qwen2-VL model, good for basic testing",
        "series": "qwen2",
    },
    "qwen2-vl-7b": {
        "id": "Qwen/Qwen2-VL-7B-Instruct",
        "gpu": "A100",
        "memory": 49152,
        "description": "Medium Qwen2-VL model",
        "series": "qwen2",
    },
    # Qwen3-VL models
    "qwen3-vl-8b": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "gpu": "A100",
        "memory": 65536,
        "description": "Qwen3-VL 8B — balanced speed and quality",
        "series": "qwen3",
    },
    "qwen3-vl-235b": {
        "id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "gpu": "H100:8",
        "memory": 524288,
        "description": "Qwen3-VL 235B — state-of-the-art visual understanding",
        "series": "qwen3",
    },
    # Qwen3-Omni models (Audio + Video understanding)
    "qwen3-omni-30b-thinking": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "gpu": "A100-80GB",
        "memory": 65536,
        "description": "Qwen3-Omni 30B Thinking — audio+video with reasoning (text output only)",
        "series": "qwen3-omni",
    },
    "qwen3-omni-30b-instruct": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "gpu": "H100",
        "memory": 65536,
        "description": "Qwen3-Omni 30B Instruct — audio+video understanding",
        "series": "qwen3-omni",
    },
}


@app.cls(
    gpu="A100",
    volumes={CACHE_DIR: cache_vol},
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    memory=65536,
)
class QwenVLAnalyzer:
    """Visual-only video analyzer using Qwen2-VL or Qwen3-VL models."""

    model_name: str = modal.parameter(default="qwen3-vl-8b")

    @modal.enter()
    def setup(self):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        if self.model_name not in MODELS:
            raise ValueError(f"Model {self.model_name} not found. Available: {list(MODELS.keys())}")

        model_config = MODELS[self.model_name]
        self.model_id = model_config["id"]
        self.series = model_config["series"]

        print(f"Loading model: {self.model_id} ({model_config['description']})")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
        }
        if self.series == "qwen3":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        print(f"Model loaded on {self.model.device} ({self.model.dtype})")

    @modal.method()
    def analyze_video(
        self,
        video_url: str,
        prompt: str = "Describe this video in detail.",
        max_pixels: int = 360 * 420,
        fps: float = 1.0,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        temporal_mode: bool = False,
    ) -> Dict[str, Any]:
        """Analyze a video from a public URL.

        Args:
            video_url: Public URL of the video to analyze.
            prompt: Instruction for the model.
            max_pixels: Maximum pixels per frame (controls resolution vs speed).
            fps: Frames per second to sample from the video.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            temporal_mode: Enhance prompt to request timestamps for major events.
        """
        import requests
        import tempfile
        import cv2
        import torch
        from qwen_vl_utils import process_vision_info

        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            print(f"Video: {width}x{height}, {total_frames} frames, {video_fps:.2f} fps, {duration:.2f}s")

            if temporal_mode:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds}s"
                final_prompt = (
                    f"{prompt}\n\nThis is a {duration_str} ({duration:.1f}s) video. "
                    "Please provide timestamps (MM:SS or SS format) for major events, "
                    "scene changes, or significant actions."
                )
                print("Temporal mode: enhanced prompt with duration context")
            else:
                final_prompt = prompt

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
                        {"type": "text", "text": final_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            frames_sampled = len(video_inputs[0]) if video_inputs else 0
            print(f"Sampled {frames_sampled} frames at {fps} fps")

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            print("Generating analysis...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )

            output_text = self.processor.batch_decode(
                [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return {
                "prompt": prompt,
                "video_url": video_url,
                "analysis": output_text,
                "video_metadata": {
                    "resolution": f"{width}x{height}",
                    "total_frames": total_frames,
                    "fps": video_fps,
                    "duration_seconds": duration,
                    "processing_fps": fps,
                    "frames_sampled": frames_sampled,
                    "max_pixels": max_pixels,
                },
                "generation_config": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "model": self.model_id,
                "model_series": self.series,
            }

        finally:
            Path(video_path).unlink(missing_ok=True)


@app.cls(
    gpu="A100-80GB",
    volumes={CACHE_DIR: cache_vol},
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    memory=163840,
)
class QwenOmniAnalyzer:
    """Audio + Video analyzer using Qwen3-Omni models with chunked processing."""

    model_name: str = modal.parameter(default="qwen3-omni-30b-thinking")
    quantize_8bit: bool = modal.parameter(default=False)

    @modal.enter()
    def setup(self):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        import torch

        if self.model_name not in MODELS:
            raise ValueError(f"Model {self.model_name} not found. Available: {list(MODELS.keys())}")

        model_config = MODELS[self.model_name]
        self.model_id = model_config["id"]
        self.series = model_config["series"]

        print(f"Loading model: {self.model_id} ({model_config['description']})")
        print(f"8-bit quantization: {'Enabled' if self.quantize_8bit else 'Disabled'}")

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        model_kwargs = {
            "device_map": "auto",
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        }
        if self.quantize_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)

        # Disable audio output — text only
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
            print("Talker disabled (text output only)")

        print(f"Model loaded on {self.model.device}")

    def _extract_video_chunk(self, input_path: str, start_time: float, end_time: float, output_path: str) -> None:
        """Extract a video segment using ffmpeg (stream copy, no re-encoding)."""
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(end_time - start_time),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _process_chunk(
        self,
        video_path: str,
        chunk_start: float,
        chunk_end: float,
        prompt: str,
        max_pixels: int,
        fps: float,
        max_new_tokens: int,
        use_audio_in_video: bool,
    ) -> Dict[str, Any]:
        """Run inference on a single video chunk."""
        import torch
        from qwen_omni_utils import process_mm_info

        start_min, start_sec = int(chunk_start // 60), int(chunk_start % 60)
        end_min, end_sec = int(chunk_end // 60), int(chunk_end % 60)
        time_range = f"{start_min}:{start_sec:02d} – {end_min}:{end_sec:02d}"

        chunk_prompt = (
            f"{prompt}\n\nAnalyze this segment ({time_range}) and provide timestamps "
            "relative to the overall video timeline."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
                    {"type": "text", "text": chunk_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

        inputs = self.processor(
            text=[text],
            audio=audios,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=use_audio_in_video,
        ).to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                return_audio=False,
                use_audio_in_video=use_audio_in_video,
            )

        output_text = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"start_time": chunk_start, "end_time": chunk_end, "time_range": time_range, "analysis": output_text}

    @modal.method()
    def analyze_video(
        self,
        video_url: str,
        prompt: str = "Describe this video in detail.",
        max_pixels: int = 360 * 420,
        fps: float = 1.0,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        temporal_mode: bool = True,
        use_audio_in_video: bool = True,
        enable_chunking: bool = True,
        chunk_duration: int = 30,
        min_duration_for_chunking: int = 120,
    ) -> Dict[str, Any]:
        """Analyze a video (with audio) from a public URL using Qwen3-Omni.

        Args:
            video_url: Public URL of the video to analyze.
            prompt: Instruction for the model.
            max_pixels: Maximum pixels per frame.
            fps: Frames per second to sample.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            temporal_mode: Enhance prompt to request timestamps.
            use_audio_in_video: Extract and process audio track from video.
            enable_chunking: Split long videos into chunks to avoid OOM.
            chunk_duration: Duration of each chunk in seconds.
            min_duration_for_chunking: Videos longer than this are chunked.
        """
        import requests
        import tempfile
        import cv2
        import math
        from qwen_omni_utils import process_mm_info
        import torch

        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            print(f"Video: {width}x{height}, {total_frames} frames, {video_fps:.2f} fps, {duration:.2f}s")
            print(f"Audio: {'Enabled' if use_audio_in_video else 'Disabled'}")

            use_chunking = enable_chunking and duration > min_duration_for_chunking

            if use_chunking:
                num_chunks = math.ceil(duration / chunk_duration)
                print(f"\nChunked processing: {num_chunks} chunks of {chunk_duration}s each")
                chunk_results = []

                for i in range(num_chunks):
                    chunk_start = i * chunk_duration
                    chunk_end = min((i + 1) * chunk_duration, duration)
                    chunk_path = f"/tmp/chunk_{i}.mp4"
                    print(f"\n--- Chunk {i+1}/{num_chunks} ({chunk_start:.1f}s – {chunk_end:.1f}s) ---")

                    self._extract_video_chunk(video_path, chunk_start, chunk_end, chunk_path)
                    try:
                        result = self._process_chunk(
                            video_path=chunk_path,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            prompt=prompt,
                            max_pixels=max_pixels,
                            fps=fps,
                            max_new_tokens=max_new_tokens,
                            use_audio_in_video=use_audio_in_video,
                        )
                        chunk_results.append(result)
                        print(f"Chunk {i+1}/{num_chunks} complete")
                    finally:
                        Path(chunk_path).unlink(missing_ok=True)

                combined_analysis = f"=== VIDEO ANALYSIS ({duration:.1f}s, {num_chunks} chunks) ===\n\n"
                for i, r in enumerate(chunk_results):
                    combined_analysis += f"\n### Segment {i+1}: {r['time_range']}\n\n{r['analysis']}\n\n{'—'*60}\n"

                return {
                    "prompt": prompt,
                    "video_url": video_url,
                    "analysis": combined_analysis,
                    "chunked": True,
                    "num_chunks": num_chunks,
                    "chunk_duration": chunk_duration,
                    "chunks": chunk_results,
                    "video_metadata": {
                        "resolution": f"{width}x{height}",
                        "total_frames": total_frames,
                        "fps": video_fps,
                        "duration_seconds": duration,
                        "processing_fps": fps,
                        "max_pixels": max_pixels,
                        "audio_enabled": use_audio_in_video,
                    },
                    "generation_config": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k},
                    "model": self.model_id,
                    "model_series": self.series,
                }

            else:
                # Single-pass for short videos
                if temporal_mode:
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    duration_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds}s"
                    final_prompt = (
                        f"{prompt}\n\nThis is a {duration_str} ({duration:.1f}s) video. "
                        "Please provide timestamps (MM:SS or SS format) for major events."
                    )
                else:
                    final_prompt = prompt

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
                            {"type": "text", "text": final_prompt},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)

                frames_sampled = len(videos[0]) if videos else 0
                has_audio = audios is not None and len(audios) > 0
                print(f"Sampled {frames_sampled} frames | Audio extracted: {has_audio}")

                inputs = self.processor(
                    text=[text],
                    audio=audios,
                    images=images,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                    use_audio_in_video=use_audio_in_video,
                ).to(self.model.device).to(self.model.dtype)

                print("Generating analysis...")
                with torch.no_grad():
                    text_ids, _ = self.model.generate(
                        **inputs,
                        thinker_return_dict_in_generate=True,
                        thinker_max_new_tokens=max_new_tokens,
                        thinker_do_sample=False,
                        return_audio=False,
                        use_audio_in_video=use_audio_in_video,
                    )

                output_text = self.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                return {
                    "prompt": prompt,
                    "video_url": video_url,
                    "analysis": output_text,
                    "chunked": False,
                    "video_metadata": {
                        "resolution": f"{width}x{height}",
                        "total_frames": total_frames,
                        "fps": video_fps,
                        "duration_seconds": duration,
                        "processing_fps": fps,
                        "frames_sampled": frames_sampled,
                        "max_pixels": max_pixels,
                        "audio_enabled": use_audio_in_video,
                        "audio_extracted": has_audio,
                    },
                    "generation_config": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k},
                    "model": self.model_id,
                    "model_series": self.series,
                }

        finally:
            Path(video_path).unlink(missing_ok=True)


@app.local_entrypoint()
def main(
    video_url: Optional[str] = None,
    prompt: Optional[str] = None,
    model: str = "qwen3-vl-8b",
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
    temperature: float = 0.7,
    top_p: float = 0.9,
    quantize_8bit: bool = False,
):
    """Run video analysis from the command line.

    Examples:
        # Visual analysis with Qwen3-VL (default)
        modal run analyze_video.py

        # Audio + visual with Qwen3-Omni
        modal run analyze_video.py --model qwen3-omni-30b-thinking

        # Qwen3-Omni with 8-bit quantization (lower memory)
        modal run analyze_video.py --model qwen3-omni-30b-thinking --quantize-8bit

        # Largest visual model (needs 8× H100)
        modal run analyze_video.py --model qwen3-vl-235b

        # Custom video
        modal run analyze_video.py \\
          --model qwen3-omni-30b-thinking \\
          --video-url "https://example.com/video.mp4" \\
          --prompt "Describe what you see and hear in detail" \\
          --fps 2.0
    """
    if video_url is None:
        video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space.mp4"
        print(f"Using default test video: {video_url}")

    if prompt is None:
        prompt = (
            "Describe what is happening in this video in detail. "
            "Transcribe step by step so the output can be used as lecture notes."
        )

    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model]

    print("\n" + "=" * 70)
    print("QWEN VIDEO ANALYZER")
    print("=" * 70)
    print(f"Model    : {model}")
    print(f"Config   : {model_config['description']}")
    print(f"GPU      : {model_config['gpu']}")
    print(f"Memory   : {model_config['memory'] / 1024:.1f} GB")
    print(f"Series   : {model_config['series'].upper()}")
    print("=" * 70 + "\n")

    if model_config["series"] == "qwen3-omni":
        analyzer = QwenOmniAnalyzer(model_name=model, quantize_8bit=quantize_8bit)
        result = analyzer.analyze_video.remote(
            video_url=video_url,
            prompt=prompt,
            fps=fps,
            max_pixels=max_pixels,
            temperature=temperature,
            top_p=top_p,
            use_audio_in_video=True,
        )
    else:
        analyzer = QwenVLAnalyzer(model_name=model)
        result = analyzer.analyze_video.remote(
            video_url=video_url,
            prompt=prompt,
            fps=fps,
            max_pixels=max_pixels,
            temperature=temperature,
            top_p=top_p,
        )

    output_file = f"video_analysis_{model}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Model    : {result['model']} ({result['model_series'].upper()})")
    print(f"Video    : {result['video_url']}")
    print(f"Duration : {result['video_metadata']['duration_seconds']:.2f}s")
    print(f"Mode     : {'CHUNKED (' + str(result['num_chunks']) + ' chunks)' if result.get('chunked') else 'SINGLE PASS'}")
    print(f"\nPrompt: {result['prompt']}\n")
    print("Analysis:")
    print("-" * 70)
    print(result["analysis"])
    print("-" * 70)
    print(f"\nFull results saved to: {output_file}")


@app.function(
    gpu="A100",
    volumes={CACHE_DIR: cache_vol},
    timeout=3600,
    memory=65536,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web endpoint for the video analyzer."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class VideoAnalysisRequest(BaseModel):
        video_url: str
        prompt: str = "Describe this video in detail."
        model: str = "qwen3-vl-8b"
        fps: float = 1.0
        max_pixels: int = 360 * 420
        temperature: float = 0.7
        top_p: float = 0.9
        temporal_mode: bool = True

    web_app = FastAPI(
        title="Qwen Video Analyzer",
        description="Analyze videos using Qwen2-VL, Qwen3-VL, and Qwen3-Omni models",
        version="1.0.0",
    )

    @web_app.get("/")
    async def root():
        return {
            "message": "Qwen Video Analyzer API",
            "available_models": list(MODELS.keys()),
            "model_categories": {
                "qwen2_vl": [k for k, v in MODELS.items() if v["series"] == "qwen2"],
                "qwen3_vl": [k for k, v in MODELS.items() if v["series"] == "qwen3"],
                "qwen3_omni": [k for k, v in MODELS.items() if v["series"] == "qwen3-omni"],
            },
            "endpoints": {"POST /analyze": "Analyze a video", "GET /models": "List models", "GET /health": "Health check"},
        }

    @web_app.get("/models")
    async def list_models():
        return {
            "models": {
                name: {"id": cfg["id"], "description": cfg["description"], "series": cfg["series"], "gpu": cfg["gpu"]}
                for name, cfg in MODELS.items()
            }
        }

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "available_models": list(MODELS.keys())}

    @web_app.post("/analyze")
    async def analyze_video_endpoint(request: VideoAnalysisRequest):
        if request.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}. Available: {list(MODELS.keys())}")
        try:
            cfg = MODELS[request.model]
            if cfg["series"] == "qwen3-omni":
                analyzer = QwenOmniAnalyzer(model_name=request.model)
                return analyzer.analyze_video.remote(
                    video_url=request.video_url,
                    prompt=request.prompt,
                    fps=request.fps,
                    max_pixels=request.max_pixels,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    temporal_mode=request.temporal_mode,
                    use_audio_in_video=True,
                )
            else:
                analyzer = QwenVLAnalyzer(model_name=request.model)
                return analyzer.analyze_video.remote(
                    video_url=request.video_url,
                    prompt=request.prompt,
                    fps=request.fps,
                    max_pixels=request.max_pixels,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    temporal_mode=request.temporal_mode,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app
