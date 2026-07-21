"""
check_setup.py — verify TalkFace Brain package is ready to run.
Run from this folder:  python check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ok = True


def check(name: str, cond: bool, hint: str = "") -> None:
    global ok
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        ok = False
        if hint:
            print(f"         → {hint}")


def main() -> int:
    print("=" * 60)
    print("TalkFace Brain — setup check")
    print("=" * 60)
    print(f"Root: {ROOT}\n")

    print("Python packages")
    for mod in [
        "numpy",
        "soundfile",
        "sounddevice",
        "torch",
        "torchaudio",
        "transformers",
        "onnxruntime",
        "huggingface_hub",
    ]:
        try:
            __import__(mod)
            check(mod, True)
        except ImportError:
            check(mod, False, f"pip install {mod}")

    try:
        import torchcrepe  # noqa: F401
        check("torchcrepe", True)
    except ImportError:
        check("torchcrepe", False, "pip install torchcrepe")

    try:
        import parler_tts  # noqa: F401
        check("parler_tts", True)
    except ImportError:
        check(
            "parler_tts",
            False,
            "pip install git+https://github.com/huggingface/parler-tts.git",
        )

    print("\nCode files")
    for f in [
        "orchestrator_brain.py",
        "brain_inference.py",
        "brain_model.py",
        "audio2emotion.py",
        "wav2arkit.py",
        "parler_voice.py",
        "emotion_map.py",
        "emotion_manager.py",
        "prosody_gpu.py",
        "blender_receiver.py",
        "orchestrator.py",
    ]:
        check(f, (ROOT / f).is_file())

    print("\nModels")
    parler = ROOT / "models" / "parler-tts-mini-v1"
    check(
        "parler-tts-mini-v1",
        (parler / "config.json").is_file(),
        "huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models/parler-tts-mini-v1",
    )

    w2a = ROOT / "models" / "wav2arkit_cpu" / "wav2arkit_cpu.onnx"
    check(str(w2a.relative_to(ROOT)), w2a.is_file(), "see models/README.md")

    a2e = ROOT / "models" / "audio2emotion_v2.2" / "network.onnx"
    check(
        "audio2emotion network.onnx",
        a2e.is_file() and a2e.stat().st_size > 1_000_000_000,
        "python audio2emotion.py --download  (HF login + license)",
    )

    brain = ROOT / "models" / "brain"
    for name in ("shared_encoder.pt", "face_head.pt"):
        p = brain / name
        check(f"brain/{name}", p.is_file(), "export or copy trained modules")
    adapter = brain / "character_adapter.pt"
    adapter2 = brain / "adapter.pt"
    check(
        "brain/character_adapter.pt|adapter.pt",
        adapter.is_file() or adapter2.is_file(),
        "copy adapter weights",
    )
    check(
        "brain/hubert-base-ls960/config.json",
        (brain / "hubert-base-ls960" / "config.json").is_file(),
        "huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960",
    )
    check(
        "brain/emotion_stats.json",
        (brain / "emotion_stats.json").is_file(),
        "fallback averages (should ship with this package)",
    )

    print("\nGPU")
    try:
        import torch

        cuda = torch.cuda.is_available()
        check("CUDA available", cuda, "CPU works but will be slow")
        if cuda:
            print(f"         device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        check("torch CUDA probe", False, str(e))

    print("\n" + "=" * 60)
    if ok:
        print("READY — next:")
        print("  1. Blender: open face.blend, run blender_receiver.py")
        print("  2. python orchestrator_brain.py")
        return 0
    print("NOT READY — fix FAIL items above, then re-run check_setup.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
