# TalkFace Brain

Drive a **Faceit / ARKit-style** character in **Blender** with:

**JSON speech → Parler voice → realistic lip-sync + emotion → live 3D face**

This package is the **Brain hybrid pipeline** (not the classic Rhubarb-only or agents-only demos).

---

## What you get

| Capability | Implementation |
|------------|----------------|
| **Speech** | Parler-TTS Mini (emotion-styled voice) |
| **Lip sync** | **wav2arkit** (continuous ARKit mouth @ 30 fps) |
| **Emotion from audio** | **NVIDIA Audio2Emotion-v2.2** → `emotion_26d` (26 floats) |
| **Upper face** | Brain model upper head + `emotion_map` presets |
| **Playback** | Audio + dual UDP (viseme mouth / emotion brows-eyes) |
| **Blender** | `blender_receiver.py` applies shape keys in real time |

Default mode is **hybrid**:

- Mouth = wav2arkit (best lip-sync)  
- Emotion conditioning = Audio2Emotion on the same WAV (not label averages)  
- Upper face = Brain + readable emotion presets  

---

## Pipeline

```
JSON sentences  { text, emotion, intensity }
        │
        ▼
┌─────────────────────┐
│  Parler-TTS Mini    │  → temp/*.wav
└──────────┬──────────┘
           │
     ┌─────┴──────┬─────────────────┐
     ▼            ▼                 ▼
 wav2arkit   Audio2Emotion-v2.2   Brain (HuBERT + face)
  mouth       emotion_26d [26]    upper + optional full face
     │            │                 │
     └────────────┴────────┬────────┘
                           ▼
              Play audio + UDP :9001
                           │
                           ▼
              blender_receiver.py → mesh
```

---

## Repository layout (this folder)

| Path | Purpose |
|------|---------|
| `orchestrator_brain.py` | **Main entry** — paste JSON, run full pipeline |
| `orchestrator.py` | Shared UDP / config helpers (imported by Brain orchestrator) |
| `brain_inference.py` | Load Brain modules, `run_brain()`, emotion_26d resolution |
| `brain_model.py` | Network architecture (HuBERT, dual heads, ARKit scatter) |
| `audio2emotion.py` | NVIDIA A2E ONNX → `emotion_26d` |
| `wav2arkit.py` | Audio → 52 ARKit mouth blendshapes |
| `parler_voice.py` | Parler-TTS wrapper (local `models/parler-tts-mini-v1`) |
| `prosody_gpu.py` | Pitch / energy / rate for Brain |
| `emotion_map.py` | Upper-face emotion presets for Blender |
| `emotion_manager.py` | Fallback averages from `emotion_stats.json` |
| `blender_receiver.py` | **Run inside Blender** — UDP listener |
| `export_brain_modules.py` | Split monolithic `brain.pt` → 3 modules |
| `extract_emotion_stats.py` | Optional: rebuild stats from train.parquet |
| `check_setup.py` | Verify installs and model paths |
| `download_assets.ps1` | Windows download helper for HF models |
| `requirements.txt` | Python dependencies |
| `face.blend` | Example Blender scene |
| `models/` | Weights go here (see `models/README.md`) |
| `.gitignore` | Excludes multi-GB weights and temp files |

Large weights are **not** committed to git. Download them after clone.

---

## Requirements

- **Python 3.10+** (system Python — **not** Blender’s bundled Python)
- **Blender** with ARKit / Faceit-style shape keys (`face.blend` included)
- **Windows** primary; Linux should work with path tweaks
- **GPU + CUDA strongly recommended** (Parler + Brain + crepe)
- **~8–12 GB free disk** for models; **~8 GB+ VRAM** ideal

---

## Setup

### 1. Clone / copy this folder

```powershell
cd git_upload3
```

### 2. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install git+https://github.com/huggingface/parler-tts.git
```

Install **PyTorch + torchaudio** for your CUDA/CPU build:  
https://pytorch.org/get-started/locally/

Example (adjust CUDA version as needed):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Optional GPU ONNX for Audio2Emotion:

```powershell
pip install onnxruntime-gpu
```

### 3. Download models

```powershell
# Login once (needed for gated Audio2Emotion)
huggingface-cli login
# Accept license in browser:
#   https://huggingface.co/nvidia/Audio2Emotion-v2.2

.\download_assets.ps1
```

**Brain trained modules** (required for upper face / full Brain):

Copy into `models/brain/`:

```text
shared_encoder.pt
face_head.pt
character_adapter.pt   (or adapter.pt)
hubert-base-ls960/     (if not downloaded by the script)
```

If you only have a monolithic checkpoint:

```powershell
python export_brain_modules.py path\to\brain_latest.pt
```

Details: **`models/README.md`**.

### 4. Verify

```powershell
python check_setup.py
```

All critical items should be `OK`.

---

## Run

### A. Blender (first)

1. Open `face.blend` (or your Faceit character).  
2. Scripting workspace → open `blender_receiver.py` → **Run Script**.  
3. Or: Text Editor → run operator to start the stream receiver (see script header).  
4. Leave Blender running; it listens on **UDP `127.0.0.1:9001`**.

### B. Python orchestrator

```powershell
python orchestrator_brain.py
```

Wait until models load, press **Enter** when Blender is ready, then paste JSON and a **blank line**.

---

## JSON input format

```json
{
  "sentences": [
    {"text": "Hello.", "emotion": "happy", "intensity": 0.9},
    {"text": "I am sorry about that.", "emotion": "sad", "intensity": 0.85},
    {"text": "We need this done now.", "emotion": "angry", "intensity": 0.9}
  ]
}
```

| Field | Meaning |
|-------|---------|
| `text` | Words to speak (Parler TTS) |
| `emotion` | Voice style + face presets + A2E preferred blend |
| `intensity` | `0.0`–`1.0` strength |

### Example conversation pack

```json
{
  "sentences": [
    {"text": "Hi there. Thanks for joining me today.", "emotion": "happy", "intensity": 0.85},
    {"text": "I need to explain something important.", "emotion": "neutral", "intensity": 0.7},
    {"text": "I am worried we might miss the deadline.", "emotion": "fear", "intensity": 0.8},
    {"text": "That was a complete disaster.", "emotion": "angry", "intensity": 0.9},
    {"text": "I am really sorry. That was my fault.", "emotion": "sad", "intensity": 0.9},
    {"text": "Wait. Did you hear that?", "emotion": "surprised", "intensity": 0.95},
    {"text": "Okay. Let me think for a second.", "emotion": "thinking", "intensity": 0.7},
    {"text": "We can still fix this. I believe in the team.", "emotion": "encouraging", "intensity": 0.85},
    {"text": "Thank you for listening. Talk soon.", "emotion": "calm", "intensity": 0.75}
  ]
}
```

### Single word, multiple emotions (clear face/voice A/B)

```json
{
  "sentences": [
    {"text": "Hello.", "emotion": "neutral", "intensity": 0.6},
    {"text": "Hello.", "emotion": "happy", "intensity": 0.9},
    {"text": "Hello.", "emotion": "sad", "intensity": 0.9},
    {"text": "Hello.", "emotion": "angry", "intensity": 0.9},
    {"text": "Hello.", "emotion": "surprised", "intensity": 0.95}
  ]
}
```

Type `quit` to exit the orchestrator.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MOUTH_ENGINE` | `hybrid` | `hybrid` = wav2arkit lips + Brain upper; `brain` = pure Brain mouth; `wav2arkit` = classic mouth only |
| `EMOTION_26D_SOURCE` | `a2e` | `a2e` = Audio2Emotion from WAV; `stats` = label averages fallback |
| `A2E_PREFERRED_STRENGTH` | `0.35` | Blend JSON emotion into A2E (`0` = audio only) |
| `LIP_DELAY_MS` | `150` | Hold lips so they don’t lead audio (raise if still early) |
| `LIP_MODEL_LEAD_MS` | `30` | Extra hold for model anticipation |
| `PARLER_TORCH_COMPILE` | `0` | Set `1` only if compile helps on your GPU |

Example:

```powershell
$env:MOUTH_ENGINE="hybrid"
$env:LIP_DELAY_MS="180"
python orchestrator_brain.py
```

---

## Emotions

### Audio2Emotion (from speech) — **6 classes**

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`  

Packed into **26-D** (`16` implicit + `10` explicit) for Brain, matching the teacher layout used in training.

### App / JSON labels (voice + face presets)

Including: `neutral`, `happy`, `sad`, `angry`, `surprised`, `fearful`, `disgusted`, `thinking`, `calm`, and dataset-style tags such as `apologetic`, `assertive`, `concerned`, `encouraging` (mapped to nearest A2E class + presets).

Brain training used **12 dataset labels**; live A2E only **detects 6** from audio. Extra labels still affect **Parler voice** and **upper-face presets**.

---

## Performance notes

| Stage | Typical cost |
|-------|----------------|
| First Parler / Brain / A2E load | Several seconds–tens of seconds (once) |
| Parler TTS per sentence | Scales with text length (often the longest step) |
| Audio2Emotion | ~0.2–0.5 s/clip after warmup (CPU ONNX; faster with GPU EP) |
| wav2arkit | ~0.3–1 s short clip |
| Brain HuBERT + face | ~0.2–1 s short clip on GPU |

**Why some lines feel much slower**

- Longer text → longer Parler generation  
- First run after idle → GPU/CUDA warmup  
- Missing GPU → everything on CPU  
- **Bilingual / code-mixed / romanized non-English** text → Parler may generate longer audio or struggle (more tokens / odd phonetics) → **much longer TTS**, then longer lip+emotion jobs  
- Loading A2E + Brain + Parler in one process → VRAM pressure can slow subsequent steps  

The pipeline is **sequential per sentence**: TTS finishes → lips + emotion → play. No face until prep is done.

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `check_setup` FAIL on models | Run `download_assets.ps1`; copy Brain `.pt` files |
| A2E download 401 | `huggingface-cli login` + accept NVIDIA license page |
| No face motion | Blender receiver running? UDP port 9001 free? |
| Lips ahead of audio | `$env:LIP_DELAY_MS="200"` |
| Mouth sealed / downward | Hybrid mouth uses wav2arkit; ensure `MOUTH_ENGINE=hybrid` |
| Emotion flat | Raise intensity; A2E model present; `EMOTION_MAP` blend in orchestrator |
| CUDA OOM | Close other GPU apps; run shorter sentences; CPU onnxruntime for A2E |
| Very slow first sentence | Normal cold start; second sentence should be faster |

---

## License / third-party models

You must comply with each model’s license:

- [Parler-TTS](https://huggingface.co/parler-tts/parler-tts-mini-v1)  
- [wav2arkit](https://huggingface.co/myned-ai/wav2arkit_cpu)  
- [NVIDIA Audio2Emotion-v2.2](https://huggingface.co/nvidia/Audio2Emotion-v2.2) (Audio2Face project terms; gated)  
- [HuBERT](https://huggingface.co/facebook/hubert-base-ls960)  
- Your own Brain checkpoint license  

This repo ships **code + small configs** (`emotion_stats.json`, A2E json sidecars). Weights are downloaded by you.

---

## Quick start checklist

1. [ ] `pip install -r requirements.txt` + Parler + PyTorch  
2. [ ] `huggingface-cli login` + A2E license  
3. [ ] `.\download_assets.ps1`  
4. [ ] Copy Brain `*.pt` (+ HuBERT if needed)  
5. [ ] `python check_setup.py`  
6. [ ] Blender: `blender_receiver.py`  
7. [ ] `python orchestrator_brain.py`  
8. [ ] Paste JSON → blank line → watch the face  

---

## Support paths

- Models & paths → `models/README.md`  
- Planned work → `upcoming_features.txt`  
- Smoke test (if Brain weights present) → `python test_brain_pipeline.py`  
