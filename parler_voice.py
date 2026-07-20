"""
parler_voice.py

Parler-TTS Mini v1 helper for TalkFace.
Emotion-aware speech from text + style descriptions.

Model downloads into: models/parler-tts-mini-v1/ (~880 MB–1.2 GB, project folder)

Named parler_voice.py so it does NOT shadow the installed package `parler_tts`.
Import this module as:
    from parler_voice import generate_speech, build_voice_style, load_parler
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoTokenizer

try:
    from parler_tts import ParlerTTSForConditionalGeneration
except ImportError:  # pragma: no cover
    ParlerTTSForConditionalGeneration = None  # type: ignore

# HuggingFace repo id (used only when downloading into the project models folder)
HF_REPO_ID = "parler-tts/parler-tts-mini-v1"

# Local project folder — all Parler weights live HERE, not in ~/.cache/huggingface
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_DIR = MODELS_DIR / "parler-tts-mini-v1"

QUALITY_SUFFIX = "The audio is high quality with  no background noise."

# Module-level cache (in-memory after load — not the disk "cache" folder)
_model = None
_tokenizer = None
_device: Optional[str] = None
_warmed_up = False
_filler_cache: dict[str, Tuple[str, int]] = {}  # name -> (wav_path, sample_rate)


def _model_is_complete(model_dir: Path) -> bool:
    """True if local folder has the main weight file and config (not a partial download)."""
    if not model_dir.is_dir():
        return False
    config_ok = (model_dir / "config.json").is_file()
    weights_ok = (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
        or any(model_dir.glob("model*.safetensors"))
    )
    # Incomplete HF temp files mean download did not finish
    incomplete = list(model_dir.rglob("*.incomplete"))
    return config_ok and weights_ok and not incomplete


def ensure_local_model() -> str:
    """
    Make sure models/parler-tts-mini-v1 exists with full weights.
    Downloads from HuggingFace into the project folder (not user cache).
    Returns local path string for from_pretrained().
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if _model_is_complete(LOCAL_MODEL_DIR):
        print(f"[parler_voice] Using local model: {LOCAL_MODEL_DIR}")
        return str(LOCAL_MODEL_DIR)

    print(f"[parler_voice] Model not complete at: {LOCAL_MODEL_DIR}")
    print(f"[parler_voice] Downloading {HF_REPO_ID} into project models/ folder...")
    print("[parler_voice] Size ~880MB–1.2GB. This is a one-time download.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the model.\n"
            "  pip install huggingface_hub"
        ) from e

    # local_dir = project folder with a clear name (no user-profile cache)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    if not _model_is_complete(LOCAL_MODEL_DIR):
        raise RuntimeError(
            f"Download finished but model looks incomplete under {LOCAL_MODEL_DIR}.\n"
            "Delete that folder and try again when your network is stable."
        )

    print(f"[parler_voice] Download complete → {LOCAL_MODEL_DIR}")
    return str(LOCAL_MODEL_DIR)


# ---------------------------------------------------------------------------
# Voice style descriptions (emotion × intensity band) — exact project mapping
# ---------------------------------------------------------------------------
VOICE_STYLES = {
    "neutral": {
        "low": (
            "Calm and clear delivery, moderate pace, "
            "natural conversational tone, no particular emphasis"
        ),
        "medium": (
            "Clear and engaging delivery, steady pace, "
            "warm conversational tone"
        ),
        "high": (
            "Very clear and composed delivery, measured pace, "
            "professional tone"
        ),
    },
    "happy": {
        "low": (
            "Warm and friendly tone, slight smile in the voice, "
            "gentle upward inflection"
        ),
        "medium": (
            "Cheerful and bright delivery, energetic pace, "
            "upbeat natural tone"
        ),
        "high": (
            "Very excited and joyful delivery, fast energetic pace, "
            "bright enthusiastic tone, rising inflection on key words"
        ),
    },
    "sad": {
        "low": (
            "Slightly subdued tone, slower pace, "
            "gentle falling intonation"
        ),
        "medium": (
            "Heavy and slow delivery, low energy, "
            "falling intonation, quiet and somber"
        ),
        "high": (
            "Very slow and heavy delivery, low pitch, long pauses, "
            "deeply somber and tired tone"
        ),
    },
    "angry": {
        "low": (
            "Firm and direct delivery, clipped words, "
            "slightly tense tone"
        ),
        "medium": (
            "Sharp and forceful delivery, fast clipped pace, "
            "hard emphasis on key words"
        ),
        "high": (
            "Very intense and forceful delivery, aggressive clipped speech, "
            "strong emphasis, tight jaw quality in voice"
        ),
    },
    "surprised": {
        "low": (
            "Slightly raised pitch, mild upward inflection, "
            "gentle breathiness"
        ),
        "medium": (
            "Noticeably raised pitch, faster pace, "
            "breathless quality, rising intonation"
        ),
        "high": (
            "Very fast and breathless delivery, high pitch, "
            "strong rising intonation, genuine astonishment in voice"
        ),
    },
    "fearful": {
        "low": (
            "Slightly tense delivery, careful pacing, "
            "quiet and cautious tone"
        ),
        "medium": (
            "Tense and quiet delivery, uneven pace, "
            "slightly shaky quality"
        ),
        "high": (
            "Very quiet and tense delivery, fast uncertain pace, "
            "shaky breathless quality, whisper-like intensity"
        ),
    },
    "disgusted": {
        "low": (
            "Flat and dry delivery, slightly slow pace, "
            "dismissive tone"
        ),
        "medium": (
            "Heavy and contemptuous tone, slow deliberate pace, "
            "strong distaste in delivery"
        ),
        "high": (
            "Very heavy and contemptuous tone, slow and deliberate, "
            "strong revulsion quality, drawn out vowels"
        ),
    },
    "sarcastic": {
        "low": (
            "Slightly dry delivery, mild exaggerated inflection, "
            "understated ironic tone"
        ),
        "medium": (
            "Clearly ironic delivery, exaggerated stress on key words, "
            "dry wit in tone"
        ),
        "high": (
            "Very exaggerated ironic delivery, strong emphasis on sarcastic words, "
            "drawn out stressed syllables, obvious deadpan quality"
        ),
    },
    "thinking": {
        "low": (
            "Slightly slower thoughtful pace, gentle hesitations, "
            "contemplative tone"
        ),
        "medium": (
            "Slow and deliberate delivery, noticeable pauses between thoughts, "
            "searching quality in voice"
        ),
        "high": (
            "Very slow and careful delivery, long thoughtful pauses, "
            "quiet and introspective tone, trailing off at end of sentences"
        ),
    },
}


def _intensity_band(intensity: float) -> str:
    """
    Intensity mapping:
      0.0 - 0.33  → low
      0.34 - 0.66 → medium
      0.67 - 1.0  → high
    """
    try:
        i = float(intensity)
    except (TypeError, ValueError):
        i = 0.5
    i = max(0.0, min(1.0, i))
    if i <= 0.33:
        return "low"
    if i <= 0.66:
        return "medium"
    return "high"


def build_voice_style(emotion: str, intensity: float) -> str:
    """
    Build a Parler voice description from emotion + intensity.
    Always appends the high-quality audio suffix.
    """
    emotion_key = (emotion or "neutral").lower().strip()
    if emotion_key not in VOICE_STYLES:
        emotion_key = "neutral"

    band = _intensity_band(intensity)
    style_core = VOICE_STYLES[emotion_key][band]
    return f"{style_core}. {QUALITY_SUFFIX}"


def load_parler():
    """
    Load Parler-TTS Mini v1 once (GPU preferred).
    Optimizations: float16 on CUDA, optional torch.compile, warmup generate.
    Returns (model, tokenizer, device).
    """
    global _model, _tokenizer, _device, _warmed_up

    if _model is not None and _tokenizer is not None and _device is not None:
        return _model, _tokenizer, _device

    if ParlerTTSForConditionalGeneration is None:
        raise ImportError(
            "parler-tts is not installed.\n"
            "Install with:\n"
            "  pip install git+https://github.com/huggingface/parler-tts.git\n"
            "  pip install transformers accelerate torch soundfile sounddevice numpy"
        )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = ensure_local_model()
    print(f"[parler_voice] Loading from: {local_path}")
    print(f"[parler_voice] Device: {_device}")

    _model = ParlerTTSForConditionalGeneration.from_pretrained(local_path)
    _tokenizer = AutoTokenizer.from_pretrained(local_path)
    _model.to(_device)
    _model.eval()

    # float16 on CUDA — less VRAM, faster inference
    if _device == "cuda":
        try:
            _model = _model.half()
            print("[parler_voice] model.half() applied (float16)")
        except Exception as e:
            print(f"[parler_voice] model.half() skipped: {e}")

    # torch.compile is OFF by default: on Windows + short utterances it often
    # makes TTS *slower* (10s+). Set PARLER_TORCH_COMPILE=1 to enable.
    import os as _os
    if _os.environ.get("PARLER_TORCH_COMPILE", "0") == "1":
        if hasattr(torch, "compile") and int(torch.__version__.split(".")[0]) >= 2:
            try:
                _model = torch.compile(_model)
                print("[parler_voice] torch.compile() applied")
            except Exception as e:
                print(f"[parler_voice] torch.compile() skipped: {e}")
    else:
        print("[parler_voice] torch.compile disabled (faster short sentences)")

    # Short warmup (1–2 words) — enough to allocate kernels without long delay
    if not _warmed_up:
        try:
            print("[parler_voice] Warmup generation (discarded)...")
            warm_style = build_voice_style("neutral", 0.5)
            desc = _tokenizer(warm_style, return_tensors="pt").input_ids.to(_device)
            prompt = _tokenizer("Hi.", return_tensors="pt").input_ids.to(_device)
            with torch.no_grad():
                _ = _model.generate(input_ids=desc, prompt_input_ids=prompt)
            _warmed_up = True
            print("[parler_voice] Warmup complete")
        except Exception as e:
            print(f"[parler_voice] Warmup skipped: {e}")

    print(f"[parler_voice] Model ready on {_device}")
    return _model, _tokenizer, _device


def generate_speech(
    text: str,
    voice_style: str,
    output_path: str,
    play_audio: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech with Parler-TTS and save a WAV file.

    Official Parler API:
      input_ids        = description (voice style)
      prompt_input_ids = text to speak

    Returns (audio_float32, sample_rate).
    """
    model, tokenizer, device = load_parler()

    description_ids = tokenizer(voice_style, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_ids,
            prompt_input_ids=prompt_ids,
        )

    audio = generation.cpu().float().numpy().squeeze().astype(np.float32)
    sample_rate = int(model.config.sampling_rate)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 16-bit PCM for Rhubarb / PocketSphinx compatibility
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    print(
        f"[parler_voice] Saved: {output_path}  "
        f"({len(audio) / sample_rate:.2f}s @ {sample_rate} Hz)"
    )

    if play_audio:
        sd.play(audio, sample_rate)
        sd.wait()

    return audio, sample_rate


def generate_speech_for_emotion(
    text: str,
    emotion: str,
    intensity: float,
    output_path: str,
    play_audio: bool = False,
) -> Tuple[np.ndarray, int]:
    """Build style from emotion/intensity, then generate."""
    style = build_voice_style(emotion, intensity)
    print(f"[parler_voice] emotion={emotion!r} intensity={float(intensity):.2f} band={_intensity_band(intensity)}")
    print(f"[parler_voice] style: {style[:140]}...")
    return generate_speech(text, style, output_path, play_audio=play_audio)


def ensure_filler_sounds(temp_dir: str | Path = "temp") -> dict[str, Tuple[str, int]]:
    """
    Pre-generate short thinking fillers (play while waiting on LLM).
    Returns dict name -> (wav_path, sample_rate). Cached after first call.
    """
    global _filler_cache
    if _filler_cache:
        return _filler_cache

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    fillers = {
        "hmm": ("Hmm...", "thinking", 0.5),
        "let_me_think": ("Let me think...", "thinking", 0.6),
    }

    print("[parler_voice] Pre-generating filler / thinking sounds...")
    for name, (text, emotion, intensity) in fillers.items():
        path = str(temp_dir / f"filler_{name}.wav")
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            # Reuse existing file; still need sample rate
            data, sr = sf.read(path, dtype="float32")
            _filler_cache[name] = (path, int(sr))
            print(f"[parler_voice] Filler cached on disk: {path}")
            continue
        audio, sr = generate_speech_for_emotion(
            text=text,
            emotion=emotion,
            intensity=intensity,
            output_path=path,
            play_audio=False,
        )
        _filler_cache[name] = (path, sr)

    return _filler_cache


def play_filler(name: str = "hmm") -> None:
    """Play a pre-generated filler sound (non-blocking if already loaded)."""
    cache = ensure_filler_sounds()
    if name not in cache:
        name = next(iter(cache), None)
        if name is None:
            return
    path, sr = cache[name]
    audio, _ = sf.read(path, dtype="float32")
    sd.play(audio, sr)
    # Non-blocking — caller can continue LLM work; use sd.wait() if needed


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = os.path.join("temp", "parler_test.wav")
    os.makedirs("temp", exist_ok=True)

    print("=== Parler-TTS self-test ===")
    t0 = time.time()
    generate_speech_for_emotion(
        text="Hello there. This is a short test of the talking face voice.",
        emotion="happy",
        intensity=0.7,
        output_path=out,
        play_audio=True,
    )
    print(f"Elapsed: {(time.time() - t0) * 1000:.0f} ms")
    print("Done.")
