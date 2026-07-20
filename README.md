# TalkFace — Orchestrator pipeline (JSON → speech → lips → Blender)

This package is **only** the classic **`orchestrator.py`** pipeline:

```
JSON sentences
  → Parler-TTS Mini (emotion-styled voice)
  → WAV file
  → wav2arkit (audio → ARKit mouth blendshapes @ 30 fps)
  → emotion_map (upper face: brows / eyes / cheeks)
  → UDP 127.0.0.1:9001
  → Blender (blender_receiver.py drives Faceit / ARKit shape keys)
```

**Not included here:** webcam MediaPipe, multi-agent `face_agents`, LLM agent framework (`llm_fw`).

---

## What’s in this repo (upload these)

| File | Role |
|------|------|
| `orchestrator.py` | Main pipeline: JSON in → TTS → lips → play + UDP |
| `parler_voice.py` | Parler-TTS Mini helper (load model, generate WAV) |
| `wav2arkit.py` | Audio → mouth ARKit blendshapes (ONNX) |
| `emotion_map.py` | Emotion → upper-face blendshape presets |
| `blender_receiver.py` | **Run inside Blender** — UDP listener |
| `check_setup.py` | Verify packages + model folders |
| `download_assets.ps1` | Helper to download models into `models/` |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps large models out of git |
| `face.blend` | Example Blender scene (optional ~10 MB) |
| `models/README.md` | Where to put downloaded models |
| `README.md` | This guide |

**Do not commit** (download locally instead):

| Path | Approx size | Why |
|------|-------------|-----|
| `models/parler-tts-mini-v1/` | **~3.5 GB** | TTS weights |
| `models/wav2arkit_cpu/` | **~385 MB** | Lip-sync ONNX |
| `temp/` | varies | Generated WAV/JSON |
| `rhubarb.exe` + `res/` | optional | Only if you switch to Rhubarb |
| `__pycache__/`, `*.blend1` | — | Cache / backups |

---

## Pipeline detail

### 1. Input (you paste JSON)

```json
{"sentences": [
  {"text": "Hello there, how are you today?", "emotion": "happy", "intensity": 0.75},
  {"text": "That is really sad news.", "emotion": "sad", "intensity": 0.7}
]}
```

| Field | Meaning |
|--------|---------|
| `text` | Spoken line |
| `emotion` | `neutral`, `happy`, `sad`, `angry`, `surprised`, `disgusted`, `fearful`, `sarcastic`, `thinking` |
| `intensity` | `0.0`–`1.0` |

After pasting, press **Enter on a blank line**.

### 2. TTS — Parler Mini

- Code: `parler_voice.py`
- Model folder: `models/parler-tts-mini-v1/`
- Emotion → voice style description → speech WAV in `temp/sentence_N.wav`

### 3. Lip-sync — wav2arkit (default)

- Code: `wav2arkit.py`
- Model folder: `models/wav2arkit_cpu/`
- Config: `LIPSYNC_ENGINE = "wav2arkit"` in `orchestrator.py`
- Optional: set `"rhubarb"` and install Rhubarb (see below)

### 4. Upper face — emotion_map

- Brows / eyes / cheeks from emotion + intensity
- Streamed as UDP `type: "emotion"`
- Mouth as `type: "viseme"`
- Soft `rest_pose` at end of each line

### 5. Blender

- `blender_receiver.py` listens on **UDP `127.0.0.1:9001`**
- Mesh needs ARKit-style shape keys (Faceit, etc.)

```
JSON → Parler → WAV → wav2arkit → visemes
                 ↘ emotion_map → upper face
                              → sounddevice plays audio
                              → UDP → Blender mesh
```

---

## Requirements

- **Python 3.10+** (system Python, **not** Blender’s)
- **Blender** with ARKit / Faceit shape keys (use `face.blend` or your own)
- **GPU recommended** for Parler (CUDA). CPU works but is slow
- Windows is the primary path tested here

---

## Setup (first time)

### Step 1 — Clone / copy this folder

```powershell
cd path\to\this\project
```

### Step 2 — Python packages

```powershell
pip install -r requirements.txt
```

**Parler package** (not only transformers):

```powershell
pip install git+https://github.com/huggingface/parler-tts.git
```

**PyTorch** (match your GPU/CPU):  
https://pytorch.org/get-started/locally/

Example (CUDA 11.8 — adjust for your machine):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Keep **torchaudio version aligned with torch** (mismatch breaks Parler imports).

### Step 3 — Download models into `models/` (not in git)

Create folders if needed:

```powershell
mkdir models -Force
```

#### A) Parler-TTS Mini v1 (~3.5 GB)

**HuggingFace model:**  
https://huggingface.co/parler-tts/parler-tts-mini-v1

**Download into:**

```text
models/parler-tts-mini-v1/
```

Commands:

```powershell
huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models\parler-tts-mini-v1
```

Or:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='parler-tts/parler-tts-mini-v1', local_dir=r'models/parler-tts-mini-v1', local_dir_use_symlinks=False))"
```

You should see at least:

- `models/parler-tts-mini-v1/config.json`
- `models/parler-tts-mini-v1/model.safetensors` (~3.5 GB)

No `*.incomplete` files.

#### B) wav2arkit CPU (~385 MB)

**HuggingFace model:**  
https://huggingface.co/myned-ai/wav2arkit_cpu

**Download into:**

```text
models/wav2arkit_cpu/
```

```powershell
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models\wav2arkit_cpu
```

Or:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='myned-ai/wav2arkit_cpu', local_dir=r'models/wav2arkit_cpu', local_dir_use_symlinks=False))"
```

You should see:

- `models/wav2arkit_cpu/wav2arkit_cpu.onnx`
- `models/wav2arkit_cpu/wav2arkit_cpu.onnx.data` (~380 MB)
- `models/wav2arkit_cpu/config.json`

#### C) Optional — download script

```powershell
.\download_assets.ps1
```

(Downloads the HF models into `models/` if CLI tools work.)

### Step 4 — Check setup

```powershell
python check_setup.py
```

### Step 5 — Blender receiver

1. Open `face.blend` (or your Faceit scene).
2. Open **`blender_receiver.py`** in Text Editor → **Run Script**.
3. In Blender Python console:

```python
bpy.ops.face.stream_receiver()
```

Leave Blender running. Stop later with:

```python
bpy.ops.face.stream_stop()
```

### Step 6 — Run orchestrator

```powershell
python orchestrator.py
```

1. Press Enter after the receiver is running.  
2. Paste JSON, then a **blank line**.  
3. Face should speak and move.

---

## Optional: Rhubarb instead of wav2arkit

In `orchestrator.py`:

```python
LIPSYNC_ENGINE = "rhubarb"
```

Then:

1. Download: https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest  
2. Put **`rhubarb.exe`** in the project root.  
3. Copy **`res/`** from the zip next to `rhubarb.exe`.

Default remains **`wav2arkit`** (no Rhubarb required).

---

## Expected folder layout after setup

```
your_project/
  orchestrator.py
  parler_voice.py
  wav2arkit.py
  emotion_map.py
  blender_receiver.py
  check_setup.py
  download_assets.ps1
  requirements.txt
  .gitignore
  README.md
  face.blend
  models/
    README.md
    parler-tts-mini-v1/          ← download (~3.5 GB)
      config.json
      model.safetensors
      tokenizer files...
    wav2arkit_cpu/               ← download (~385 MB)
      wav2arkit_cpu.onnx
      wav2arkit_cpu.onnx.data
      config.json
  temp/                          ← created at runtime (gitignored)
```

---

## How to run (checklist)

| Step | Action |
|------|--------|
| 1 | `pip install -r requirements.txt` + parler-tts + torch |
| 2 | Download Parler → `models/parler-tts-mini-v1/` |
| 3 | Download wav2arkit → `models/wav2arkit_cpu/` |
| 4 | Blender: run `blender_receiver.py` → `bpy.ops.face.stream_receiver()` |
| 5 | `python orchestrator.py` → paste JSON |

**Always start the Blender receiver before the orchestrator.**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Face doesn’t move | Receiver running? Console shows `CONNECTED`? Port 9001 free? |
| Parler fails | Model folder complete? `parler-tts` installed? torch + torchaudio match? |
| wav2arkit fails | `models/wav2arkit_cpu/*.onnx` + `.onnx.data` present? `onnxruntime` installed? |
| No sound | Speakers / `sounddevice` / default audio device |
| Slow TTS | Normal for Parler Mini on laptop GPU (~several seconds per line) |
| Shape keys ignored | Names must match ARKit (`jawOpen`, `mouthSmileLeft`, …) |

---

## License notes

- Your project code: choose your own license.  
- **Parler-TTS**, **wav2arkit**, and dependencies: follow their licenses on HuggingFace / GitHub.  
- Do not commit API keys or large model binaries.
