import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
        uv_version="0.10.3",
    )
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
)

app = modal.App("whisperx-transcriber", image=image)

GPU_CONFIG = "A100"
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)

# HuggingFace token — needed for speaker diarization.
# Set HF_TOKEN in your environment, or create a Modal secret named "huggingface-secret".
hf_secret = modal.Secret.from_dotenv()


@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    secrets=[hf_secret],
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=15)
class WhisperXModel:
    """WhisperX transcription with optional word-level alignment and speaker diarization."""

    @modal.enter()
    def setup(self):
        self.device = "cuda"
        self.compute_type = "float16"

    @modal.method()
    def transcribe(
        self,
        audio_url: str,
        diarize: bool = True,
        language: str = None,
        min_speakers: int = 1,
        max_speakers: int = 2,
    ):
        """Transcribe audio or video from a URL.

        Accepts any format ffmpeg supports — audio (.mp3, .wav, .m4a, .ogg) or
        video (.mp4, .mkv, .mov, .avi). Audio is extracted automatically from video.

        Args:
            audio_url: Public URL to an audio or video file.
            diarize: If True, run speaker diarization (requires HF_TOKEN).
            language: Language code (e.g. 'en', 'de', 'fr'). Auto-detected if None.
            min_speakers: Minimum expected speakers (used in diarization).
            max_speakers: Maximum expected speakers (used in diarization).

        Returns:
            Tuple of (segments, aligned_segments) or
            (segments, aligned_segments, diarized_segments) when diarize=True.
        """
        import requests
        import whisperx

        batch_size = 16

        model = whisperx.load_model(
            "large-v2",
            self.device,
            language=language,
            compute_type=self.compute_type,
            download_root=CACHE_DIR,
        )

        print(f"Downloading from: {audio_url}")
        response = requests.get(audio_url)
        response.raise_for_status()

        audio_path = "/tmp/downloaded_input"
        with open(audio_path, "wb") as f:
            f.write(response.content)

        audio = whisperx.load_audio(audio_path)

        # 1. Transcribe
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=batch_size)
        detected_language = result["language"]
        print(f"Detected language: {detected_language}")

        # 2. Word-level alignment
        print("Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=self.device)
        result_align = whisperx.align(
            result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False
        )

        if not diarize:
            return result["segments"], result_align["segments"]

        # 3. Speaker diarization
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN is required for speaker diarization. "
                "Set it via a Modal secret named 'huggingface-secret' or as an environment variable."
            )

        print(f"Diarizing (min_speakers={min_speakers}, max_speakers={max_speakers})...")
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=self.device)
        except Exception as exc:
            if "gated" in str(exc).lower() or "403" in str(exc):
                raise RuntimeError(
                    "Access denied to the pyannote diarization model. "
                    "Your HuggingFace token must belong to an account that has accepted the gated-model terms.\n\n"
                    "  1. Visit https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions\n"
                    "  2. Visit https://huggingface.co/pyannote/segmentation-3.0 and accept the conditions\n"
                    "  3. Ensure HF_TOKEN in your .env file belongs to the same account\n\n"
                    "Then re-run the command."
                ) from exc
            raise
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result_dia = whisperx.assign_word_speakers(diarize_segments, result_align)

        return result["segments"], result_align["segments"], result_dia["segments"]


@app.local_entrypoint()
def main(
    audio_url: str,
    language: str = None,
    min_speakers: int = 2,
    max_speakers: int = 3,
    no_diarize: bool = False,
):
    """Transcribe an audio or video file using WhisperX.

    Accepts any format ffmpeg supports: .mp3, .wav, .m4a, .ogg, .mp4, .mkv, .mov, etc.
    For video files, the audio track is extracted automatically.

    Args:
        audio_url: URL to the audio or video file to transcribe.
        language: Optional language code (e.g. 'en', 'es'). Auto-detected if omitted.
        min_speakers: Minimum number of speakers for diarization.
        max_speakers: Maximum number of speakers for diarization.
        no_diarize: Disable speaker diarization (faster, no HF token needed).

    Examples:
        modal run audio-transcript.py --audio-url "https://example.com/podcast.mp3"
        modal run audio-transcript.py --audio-url "https://example.com/lecture.mp4" --language en --no-diarize
        modal run audio-transcript.py --audio-url "https://example.com/interview.mp3" --min-speakers 2 --max-speakers 3
    """
    diarize = not no_diarize

    print(f"Audio URL   : {audio_url}")
    print(f"Language    : {language or 'auto-detect'}")
    print(f"Diarization : {'Enabled' if diarize else 'Disabled'}")
    if diarize:
        print(f"Speakers    : {min_speakers}–{max_speakers}")

    output = WhisperXModel().transcribe.remote(
        audio_url=audio_url,
        diarize=diarize,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if diarize:
        transcript, transcript_align, transcript_dia = output
        with open("transcript_dia.json", "w") as f:
            json.dump(transcript_dia, f, indent=2)
        print("  - transcript_dia.json")
    else:
        transcript, transcript_align = output

    with open("transcript.json", "w") as f:
        json.dump(transcript, f, indent=2)
    with open("transcript_align.json", "w") as f:
        json.dump(transcript_align, f, indent=2)

    print("\nTranscription complete! Files saved:")
    print("  - transcript.json")
    print("  - transcript_align.json")
    if diarize:
        print("  - transcript_dia.json")
