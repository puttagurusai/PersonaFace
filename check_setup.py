"""
check_setup.py — verify orchestrator.py pipeline prerequisites.
"""

import os
import sys
from pathlib import Path


def check_file(path: str, min_mb: float = 0, desc: str = "") -> bool:
    p = Path(path)
    if not p.exists():
        print(f"MISSING: {path}  ({desc})")
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb < min_mb:
        print(f"TOO SMALL: {path} ({size_mb:.1f} MB, expected > {min_mb} MB)  ({desc})")
        return False
    print(f"OK  {path} ({size_mb:.1f} MB)  {desc}")
    return True


def main():
    print("=== Orchestrator pipeline setup check ===\n")
    ok = True
    models = Path("models")

    print("--- models/ ---")
    if not models.is_dir():
        print("MISSING: models/ — create and download assets (see README)")
        ok = False
    else:
        print("OK  models/")

    # Parler
    parler = models / "parler-tts-mini-v1"
    incomplete = list(parler.rglob("*.incomplete")) if parler.is_dir() else []
    weights_ok = parler.is_dir() and (
        (parler / "model.safetensors").is_file()
        or any(parler.glob("model*.safetensors"))
    )
    if weights_ok and (parler / "config.json").is_file() and not incomplete:
        size_mb = sum(f.stat().st_size for f in parler.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"OK  models/parler-tts-mini-v1/ ({size_mb:.0f} MB)")
    elif incomplete:
        print("INCOMPLETE: models/parler-tts-mini-v1/ (*.incomplete) — re-download")
        ok = False
    else:
        print("MISSING: models/parler-tts-mini-v1/")
        print("  https://huggingface.co/parler-tts/parler-tts-mini-v1")
        print("  → download into models/parler-tts-mini-v1/")
        ok = False

    # wav2arkit
    w2a = models / "wav2arkit_cpu"
    onnx = w2a / "wav2arkit_cpu.onnx"
    data = w2a / "wav2arkit_cpu.onnx.data"
    if onnx.is_file() and data.is_file():
        print(f"OK  models/wav2arkit_cpu/ ({(onnx.stat().st_size + data.stat().st_size) / (1024*1024):.0f} MB)")
    else:
        print("MISSING: models/wav2arkit_cpu/ (need .onnx + .onnx.data)")
        print("  https://huggingface.co/myned-ai/wav2arkit_cpu")
        print("  → download into models/wav2arkit_cpu/")
        ok = False

    print("\n--- Python packages ---")
    packages = {
        "sounddevice": "playback",
        "soundfile": "WAV I/O",
        "numpy": "arrays",
        "torch": "Parler",
        "transformers": "HF",
        "huggingface_hub": "downloads",
        "onnxruntime": "wav2arkit",
        "scipy": "resample",
    }
    for mod, desc in packages.items():
        try:
            __import__(mod)
            print(f"OK  {mod}  ({desc})")
        except ImportError:
            print(f"MISSING: {mod}  ({desc})")
            ok = False

    print("\n--- Parler package ---")
    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
        print("OK  parler_tts package")
    except Exception as e:
        print(f"MISSING: parler_tts ({e})")
        print("  pip install git+https://github.com/huggingface/parler-tts.git")
        ok = False

    try:
        import parler_voice  # noqa: F401
        print(f"OK  parler_voice.py → {getattr(parler_voice, 'LOCAL_MODEL_DIR', '?')}")
    except Exception as e:
        print(f"FAIL parler_voice: {e}")
        ok = False

    try:
        import wav2arkit  # noqa: F401
        print("OK  wav2arkit.py")
    except Exception as e:
        print(f"FAIL wav2arkit: {e}")
        ok = False

    # Rhubarb optional
    print("\n--- Optional Rhubarb ---")
    rhubarb = "rhubarb.exe" if os.name == "nt" else "rhubarb"
    if Path(rhubarb).exists():
        print(f"OK  {rhubarb} (only needed if LIPSYNC_ENGINE=rhubarb)")
    else:
        print(f"INFO  {rhubarb} not present (OK — default is wav2arkit)")

    print("\n" + "=" * 40)
    if ok:
        print("Ready:")
        print("  1. Blender: blender_receiver.py → bpy.ops.face.stream_receiver()")
        print("  2. python orchestrator.py")
    else:
        print("Fix missing items above, then re-run check_setup.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
