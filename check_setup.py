"""
check_setup.py

Run this before orchestrator.py to verify everything is in place.
"""

import os
import sys
from pathlib import Path

def check_file(path: str, min_mb: float = 0, desc: str = "") -> bool:
    p = Path(path)
    if not p.exists():
        print(f"❌ MISSING: {path}  ({desc})")
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb < min_mb:
        print(f"⚠️  TOO SMALL: {path} ({size_mb:.1f} MB, expected > {min_mb} MB)  ({desc})")
        return False
    print(f"✅ {path} ({size_mb:.1f} MB)  {desc}")
    return True

def main():
    print("=== Pipeline Setup Check ===\n")

    ok = True

    # Models (TEMP: lower threshold because we are in JSON-paste mode for now)
    check_file("kokoro-v1.0.onnx", 200, "Main Kokoro TTS model (download if missing)")
    ok &= check_file("voices-v1.0.bin", 20, "Voice embeddings")

    # Rhubarb
    rhubarb = "rhubarb.exe" if os.name == "nt" else "rhubarb"
    ok &= check_file(rhubarb, 1, "Rhubarb lip-sync binary")

    # Python packages (basic import check)
    print("\n--- Python packages ---")
    packages = {
        "anthropic": "Claude API client",
        "kokoro_onnx": "Kokoro TTS (ONNX)",
        "sounddevice": "Audio playback",
        "soundfile": "WAV I/O",
    }
    for mod, desc in packages.items():
        try:
            __import__(mod)
            print(f"✅ {mod}  ({desc})")
        except ImportError:
            print(f"❌ {mod} not installed  ({desc})")
            ok = False

    # Rhubarb res/ (phonetic mode)
    print("\n--- Rhubarb resources ---")
    res_dir = Path("res") / "sphinx"
    if res_dir.is_dir():
        print(f"✅ res/sphinx/ present  (phonetic models)")
    else:
        print("❌ res/sphinx/ missing  (copy 'res' from the Rhubarb release next to rhubarb.exe)")
        ok = False

    # API key (optional in JSON-paste mode)
    print("\n--- Environment ---")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY is set")
    else:
        print("ℹ️  ANTHROPIC_API_KEY is NOT set (OK for JSON-paste mode)")
        print("   For live Claude later:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")

    print("\n" + "="*40)
    if ok:
        print("✅ Required checks passed. Ready to run:")
        print("   1. In Blender: open blender_receiver.py → Run Script")
        print("   2. Then: bpy.ops.face.stream_receiver()")
        print("   3. Terminal: python orchestrator.py")
    else:
        print("❌ Some items are missing or incomplete.")
        print("   See README.md (One-time asset setup) and run .\\download_assets.ps1")
        sys.exit(1)

if __name__ == "__main__":
    main()
