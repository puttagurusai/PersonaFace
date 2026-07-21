# Models directory

Weights are **not** in git (too large). Place downloads here with these exact names.

| Path | Approx. size | Purpose |
|------|--------------|---------|
| `parler-tts-mini-v1/` | ~3.5 GB | Parler-TTS Mini voice |
| `wav2arkit_cpu/` | ~0.5–1 GB | Audio → ARKit mouth (ONNX) |
| `audio2emotion_v2.2/network.onnx` | ~1.27 GB | Live `emotion_26d` from speech |
| `brain/shared_encoder.pt` | varies | HuBERT + fusion encoder |
| `brain/face_head.pt` | varies | Dual face heads (ARKit 52) |
| `brain/character_adapter.pt` | small | Residual adapter |
| `brain/hubert-base-ls960/` | ~0.4 GB | Local HuBERT weights |
| `brain/emotion_stats.json` | small | **Fallback** if A2E missing (included) |
| `brain/emotion_26d_layout.json` | small | Pack A2E 6-class → 26-D (included) |

## Quick download (recommended)

From the project root (this folder):

```powershell
# Windows
.\download_assets.ps1
```

Or follow step-by-step in the main **README.md**.

## Manual downloads

### Parler-TTS Mini

```powershell
huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models/parler-tts-mini-v1
```

### wav2arkit

```powershell
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models/wav2arkit_cpu
```

### Audio2Emotion-v2.2 (gated)

1. Login: `huggingface-cli login`
2. Accept license: https://huggingface.co/nvidia/Audio2Emotion-v2.2  
3. Download:

```powershell
python audio2emotion.py --download
```

This writes `models/audio2emotion_v2.2/network.onnx` (~1.27 GB).

### Brain weights + HuBERT

Copy from your training/export machine:

```text
models/brain/shared_encoder.pt
models/brain/face_head.pt
models/brain/character_adapter.pt   # or adapter.pt
models/brain/hubert-base-ls960/     # facebook/hubert-base-ls960 snapshot
```

Or export split modules from a monolithic checkpoint:

```powershell
python export_brain_modules.py path\to\brain_latest.pt
```

HuBERT local snapshot:

```powershell
huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960
```

## Disk budget (approx.)

| Component | Size |
|-----------|------|
| Parler Mini | ~3.5 GB |
| wav2arkit | ~0.5–1 GB |
| Audio2Emotion ONNX | ~1.3 GB |
| Brain + HuBERT | ~1–3 GB |
| **Total** | **~7–10 GB** |

GPU VRAM: ideally **8 GB+** free for Parler + Brain + crepe (hybrid still uses GPU for Parler/Brain).
