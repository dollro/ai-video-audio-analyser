import modal
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Configure CUDA environment
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg")
    # Install PyTorch with CUDA 12.8
    .uv_pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        index_url="https://download.pytorch.org/whl/cu128",
        uv_version="0.10.3",
    )
    .uv_pip_install(
        "fastapi[standard]",
        "uvicorn[standard]",
        "pydantic>=2.0",
        "transformers>=4.48.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.44.0",
        "qwen-vl-utils>=0.0.14",
        "qwen-omni-utils",
        "huggingface_hub[hf_transfer]",
        "safetensors",
        "tokenizers",
        "einops",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "av",
        "decord",
        "requests",
        "tqdm",
        "pyyaml",
        uv_version="0.10.3",
    )
    # Install vLLM (may adjust PyTorch internals)
    .uv_pip_install("vllm==0.13.0", uv_version="0.10.3")
    # Re-install hf_transfer after vLLM (vLLM may override huggingface_hub without the extra)
    .uv_pip_install("hf_transfer", uv_version="0.10.3")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        # Force spawn (not fork) for vLLM V1 tensor-parallel workers —
        # pynvml calls cuInit() before fork, which breaks CUDA in children.
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
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
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
    },
    "qwen2-vl-7b": {
        "id": "Qwen/Qwen2-VL-7B-Instruct",
        "gpu": "A100",
        "memory": 49152,
        "description": "Medium Qwen2-VL model",
        "series": "qwen2",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
    },
    # Qwen3-VL models
    "qwen3-vl-8b": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "gpu": "A100",
        "memory": 65536,
        "description": "Qwen3-VL 8B — balanced speed and quality",
        "series": "qwen3",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
    },
    "qwen3-vl-235b": {
        "id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "gpu": "H100:8",
        "memory": 524288,
        "description": "Qwen3-VL 235B — state-of-the-art visual understanding",
        "series": "qwen3",
        "tensor_parallel_size": 8,
        "max_model_len": 65536,
    },
    # Qwen3-Omni models (Audio + Video understanding)
    "qwen3-omni-30b-thinking": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
        "gpu": "A100-80GB:2",
        "memory": 163840,
        "description": "Qwen3-Omni 30B Thinking — audio+video with reasoning (text output only)",
        "series": "qwen3-omni",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
    },
    "qwen3-omni-30b-instruct": {
        "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        #"gpu": "H100:2",
        "gpu": "A100-80GB:2",
        "memory": 163840,
        "description": "Qwen3-Omni 30B Instruct — audio+video understanding",
        "series": "qwen3-omni",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
    },
}


hf_secret = modal.Secret.from_dotenv()


@app.cls(
    gpu="A100",
    volumes={CACHE_DIR: cache_vol},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    memory=65536,
)
class QwenVLAnalyzer:
    """Visual-only video analyzer using Qwen2-VL or Qwen3-VL models via vLLM."""

    model_name: str = modal.parameter(default="qwen3-vl-8b")

    @modal.enter()
    def setup(self):
        from vllm import LLM
        from transformers import AutoProcessor

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

        tp = model_config.get("tensor_parallel_size", 1)
        self.llm = LLM(
            model=self.model_id,
            download_dir=CACHE_DIR,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=tp,
            limit_mm_per_prompt={"image": 3, "video": 3},
            max_num_seqs=8,
            max_model_len=model_config.get("max_model_len", 32768),
            dtype="bfloat16",
            mm_encoder_tp_mode="data",
            enable_expert_parallel=tp > 1,
            enforce_eager=True,
        )
        print(f"vLLM engine ready (tp={tp})")

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
        from urllib.parse import urlparse
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        filename = Path(urlparse(video_url).path).name or "video.mp4"
        tmp_dir = tempfile.mkdtemp()
        video_path = str(Path(tmp_dir) / filename)
        with open(video_path, "wb") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

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
                    "Provide timestamps (MM:SS or SS format) for major events, "
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
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )
            frames_sampled = len(video_inputs[0]) if video_inputs else 0
            print(f"Sampled {frames_sampled} frames at {fps} fps")

            mm_data = {}
            if video_inputs:
                mm_data["video"] = video_inputs
            if image_inputs:
                mm_data["image"] = image_inputs

            vllm_input = {
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )

            print("Generating analysis...")
            outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text

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
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.cls(
    gpu="A100-80GB:2",
    volumes={CACHE_DIR: cache_vol},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    memory=163840,
)
class QwenOmniAnalyzer:
    """Audio + Video analyzer using Qwen3-Omni models via vLLM with tensor parallelism."""

    model_name: str = modal.parameter(default="qwen3-omni-30b-instruct")

    @modal.enter()
    def setup(self):
        from vllm import LLM
        from transformers import AutoProcessor

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

        tp = model_config.get("tensor_parallel_size", 2)
        self.llm = LLM(
            model=self.model_id,
            download_dir=CACHE_DIR,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=tp,
            limit_mm_per_prompt={"image": 3, "video": 3, "audio": 3},
            max_num_seqs=8,
            max_model_len=model_config.get("max_model_len", 32768),
            seed=1234,
            dtype="bfloat16",
            enforce_eager=True,
        )
        print(f"vLLM engine ready (tp={tp})")

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

    def _prepare_chunk_input(
        self,
        video_path: str,
        chunk_start: float,
        chunk_end: float,
        prompt: str,
        max_pixels: int,
        fps: float,
        use_audio_in_video: bool,
        original_video_path: str = "",
    ) -> tuple:
        """Prepare a vLLM input dict for a single video chunk.

        Returns (vllm_input_dict, time_range_str).
        """
        from qwen_omni_utils import process_mm_info

        start_min, start_sec = int(chunk_start // 60), int(chunk_start % 60)
        end_min, end_sec = int(chunk_end // 60), int(chunk_end % 60)
        time_range = f"{start_min}:{start_sec:02d} – {end_min}:{end_sec:02d}"

        chunk_prompt = (
            f"{prompt}\n\nAnalyze this segment ({time_range}) and provide timestamps "
            "relative to the overall video timeline."
        )

        # Extract audio to .wav from the original video (chunks may lack audio streams)
        # librosa/soundfile also can't decode .mp4 containers directly
        audio_path = None
        if use_audio_in_video:
            import subprocess
            audio_src = original_video_path or video_path
            audio_path = video_path.rsplit(".", 1)[0] + ".wav"
            cmd = ["ffmpeg", "-y", "-ss", str(chunk_start),
                   "-i", audio_src, "-t", str(chunk_end - chunk_start),
                   "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                   audio_path]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"Warning: audio extraction failed for chunk {chunk_start:.0f}s–{chunk_end:.0f}s: "
                      f"{result.stderr.decode()[-300:]}")
                audio_path = None

        content = [
            {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
        ]
        if audio_path:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": chunk_prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        mm_data = {}
        if videos:
            mm_data["video"] = videos
        if images:
            mm_data["image"] = images
        if audios:
            mm_data["audio"] = audios

        vllm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
        }

        return vllm_input, time_range

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
            temperature: Sampling temperature (default 0.6 per upstream recommendation).
            top_p: Top-p sampling parameter (default 0.95 per upstream recommendation).
            top_k: Top-k sampling parameter (default 20 per upstream recommendation).
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
        from urllib.parse import urlparse
        from vllm import SamplingParams
        from qwen_omni_utils import process_mm_info

        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()

        filename = Path(urlparse(video_url).path).name or "video.mp4"
        tmp_dir = tempfile.mkdtemp()
        video_path = str(Path(tmp_dir) / filename)
        with open(video_path, "wb") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / video_fps if video_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            print(f"Video: {width}x{height}, {total_frames} frames, {video_fps:.2f} fps, {duration:.2f}s")

            # Probe for audio streams — skip audio processing if none exist
            if use_audio_in_video:
                import subprocess
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
                    capture_output=True, text=True,
                )
                if not probe.stdout.strip():
                    print("Audio: No audio stream found in video, disabling audio processing")
                    use_audio_in_video = False
                else:
                    print("Audio: Enabled (audio stream detected)")
            else:
                print("Audio: Disabled")

            use_chunking = enable_chunking and duration > min_duration_for_chunking

            if use_chunking:
                num_chunks = math.ceil(duration / chunk_duration)
                print(f"\nChunked processing: {num_chunks} chunks of {chunk_duration}s each")

                # Phase 1: Extract all chunks with ffmpeg
                chunk_paths = []
                for i in range(num_chunks):
                    chunk_start = i * chunk_duration
                    chunk_end = min((i + 1) * chunk_duration, duration)
                    chunk_path = f"/tmp/chunk_{i}.mp4"
                    print(f"Extracting chunk {i+1}/{num_chunks} ({chunk_start:.1f}s – {chunk_end:.1f}s)")
                    self._extract_video_chunk(video_path, chunk_start, chunk_end, chunk_path)
                    chunk_paths.append((chunk_path, chunk_start, chunk_end))

                # Phase 2: Prepare vLLM inputs for each chunk
                vllm_inputs = []
                time_ranges = []
                for chunk_path, chunk_start, chunk_end in chunk_paths:
                    vllm_input, time_range = self._prepare_chunk_input(
                        chunk_path, chunk_start, chunk_end, prompt,
                        max_pixels, fps, use_audio_in_video,
                        original_video_path=video_path,
                    )
                    vllm_inputs.append(vllm_input)
                    time_ranges.append(time_range)

                # Phase 3: Batch inference — vLLM processes all chunks concurrently
                print(f"\nBatch generating {num_chunks} chunks...")
                outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)

                # Collect results
                chunk_results = []
                for i, (output, (_, chunk_start, chunk_end)) in enumerate(zip(outputs, chunk_paths)):
                    chunk_results.append({
                        "start_time": chunk_start,
                        "end_time": chunk_end,
                        "time_range": time_ranges[i],
                        "analysis": output.outputs[0].text,
                    })
                    print(f"Chunk {i+1}/{num_chunks} complete")

                # Cleanup chunk files
                for chunk_path, _, _ in chunk_paths:
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
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    },
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

                content = [
                    {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
                ]
                if use_audio_in_video:
                    content.append({"type": "audio", "audio": video_path})
                content.append({"type": "text", "text": final_prompt})

                messages = [{"role": "user", "content": content}]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

                frames_sampled = len(videos[0]) if videos else 0
                has_audio = audios is not None and len(audios) > 0
                print(f"Sampled {frames_sampled} frames | Audio extracted: {has_audio}")

                mm_data = {}
                if videos:
                    mm_data["video"] = videos
                if images:
                    mm_data["image"] = images
                if audios:
                    mm_data["audio"] = audios

                vllm_input = {
                    "prompt": text,
                    "multi_modal_data": mm_data,
                }

                print("Generating analysis...")
                outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)
                output_text = outputs[0].outputs[0].text

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
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    },
                    "model": self.model_id,
                    "model_series": self.series,
                }

        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.local_entrypoint()
def main(
    video_url: str = "",
    prompt: Optional[str] = None,
    model: str = "qwen3-vl-8b",
    fps: float = 1.0,
    max_pixels: int = 360 * 420,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Run video analysis from the command line.

    Examples:
        # Visual analysis with Qwen3-VL
        modal run video-analyser.py --video-url "https://example.com/video.mp4"

        # Audio + visual with Qwen3-Omni (uses 2x A100-80GB with tensor parallelism)
        modal run video-analyser.py --model qwen3-omni-30b-instruct --video-url "https://example.com/video.mp4"

        # Custom prompt
        modal run video-analyser.py \\
          --model qwen3-omni-30b-thinking \\
          --video-url "https://example.com/video.mp4" \\
          --prompt "Describe what you see and hear in detail" \\
          --fps 2.0
    """
    if not video_url:
        raise ValueError("--video-url is required. Pass a public URL to a video file.")

    if prompt is None:
        prompt = (
            "Transcribe and describe this video moment by moment in chronological order.\n\n"
            "Rules:\n"
            "- Transcribe ALL speech verbatim in quotation marks. Never paraphrase or say 'the person speaks'.\n"
            "- Describe visible actions, scenes, text on screen, and sound effects.\n"
            "- Cover every segment — do not skip or merge time ranges.\n"
            "- Do NOT summarize. Do NOT add conclusions, opinions, or an overview section.\n"
            "- Output ONLY the entries, nothing else."
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

    # Guard: QwenVLAnalyzer is hardcoded to single A100; multi-GPU VL models need their own class
    if model_config["series"] != "qwen3-omni" and model_config.get("tensor_parallel_size", 1) > 1:
        raise ValueError(
            f"Model {model} requires {model_config['gpu']} (tp={model_config['tensor_parallel_size']}), "
            f"but QwenVLAnalyzer only supports single-GPU models. "
            f"Multi-GPU VL support is not yet implemented."
        )

    if model_config["series"] == "qwen3-omni":
        analyzer = QwenOmniAnalyzer(model_name=model)
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


@app.function(timeout=3600)
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
            if cfg["series"] != "qwen3-omni" and cfg.get("tensor_parallel_size", 1) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model} requires multi-GPU ({cfg['gpu']}), not yet supported for VL models.",
                )
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
