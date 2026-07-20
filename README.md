# TalkFace

Drive a Faceit / ARKit-style character in **Blender** from spoken lines:

**JSON text → Parler speech → lip-sync + emotion → live face in Blender**

---

## Pipeline overview

```
You paste JSON sentences
        │
        ▼
┌───────────────────┐
│  Parler-TTS Mini  │  emotion-styled voice → WAV
└─────────┬─────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
┌───────────────────┐   ┌───────────────────┐
│  wav2arkit (ONNX) │   │  emotion_map      │
│  mouth / jaw @30fps│   │  brows / eyes     │
└─────────┬─────────┘   └─────────┬─────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              Play audio (speakers)
              UDP → 127.0.0.1:9001
                      │
                      ▼
              blender_receiver.py
              (inside Blender)
                      │
                      ▼
              Character face moves
```

| Stage | What it does |
|--------|----------------|
| **Input** | Short sentences with emotion + intensity |
| **TTS** | Parler-TTS Mini generates speech audio |
| **Lips** | wav2arkit turns audio into ARKit mouth shapes |
| **Emotion** | Upper face (brows, eyes, cheeks) from emotion presets |
| **Blender** | Receiver applies shape keys in real time |

---

## Repository contents

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main app — paste JSON, generate speech, stream face data |
| `parler_voice.py` | Parler-TTS helper |
| `wav2arkit.py` | Audio → ARKit mouth blendshapes |
| `emotion_map.py` | Emotion → upper-face blendshape presets |
| `blender_receiver.py` | **Run inside Blender** — listens for UDP and drives the mesh |
| `check_setup.py` | Checks that packages and models are installed |
| `download_assets.ps1` | Windows helper to download model files |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignores large models and temp files |
| `face.blend` | Example Blender scene with a face mesh |
| `models/README.md` | Where to place downloaded models |
| `upcoming_features.txt` | Planned improvements |
| `README.md` | This guide |

Large model weights are **not** stored in git (see setup below).

---

## Requirements

- **Python 3.10+** (system Python — not Blender’s built-in Python)
- **Blender** with ARKit-style shape keys (this repo includes `face.blend`, or use your own Faceit character)
- **Windows** is the primary tested platform
- **GPU with CUDA recommended** for Parler (CPU works but is much slower)

---

## Setup

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install Python packages

```powershell
pip install -r requirements.txt
pip install git+https://github.com/huggingface/parler-tts.git
```

Install **PyTorch** for your system (CUDA or CPU):  
https://pytorch.org/get-started/locally/

Example (CUDA 11.8 — change if your setup differs):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Use matching **torch** and **torchaudio** versions.

### 3. Download models (required — not in git)

Create the models folder:

```powershell
mkdir models -Force
```

#### Parler-TTS Mini v1 (~3.5 GB)

- **Page:** https://huggingface.co/parler-tts/parler-tts-mini-v1  
- **Save to:** `models/parler-tts-mini-v1/`

```powershell
huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models\parler-tts-mini-v1
```

Or:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='parler-tts/parler-tts-mini-v1', local_dir=r'models/parler-tts-mini-v1', local_dir_use_symlinks=False))"
```

You need at least `config.json` and `model.safetensors` (~3.5 GB). There must be no `*.incomplete` files.

#### wav2arkit lip-sync (~385 MB)

- **Page:** https://huggingface.co/myned-ai/wav2arkit_cpu  
- **Save to:** `models/wav2arkit_cpu/`

```powershell
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models\wav2arkit_cpu
```

Or:

```powershell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='myned-ai/wav2arkit_cpu', local_dir=r'models/wav2arkit_cpu', local_dir_use_symlinks=False))"
```

You need:

- `wav2arkit_cpu.onnx`
- `wav2arkit_cpu.onnx.data`
- `config.json`

#### Optional helper script (Windows)

```powershell
.\download_assets.ps1
```

### 4. Verify installation

```powershell
python check_setup.py
```

### 5. Start Blender

1. Open `face.blend` (or your own character with ARKit shape keys).
2. Open `blender_receiver.py` in the Text Editor → **Run Script**.
3. In the Blender Python console:

```python
bpy.ops.face.stream_receiver()
```

Leave Blender open. To stop later:

```python
bpy.ops.face.stream_stop()
```

### 6. Run the orchestrator

```powershell
python orchestrator.py
```

1. Press Enter only after the Blender receiver is running.  
2. Paste a JSON block (see below).  
3. Press **Enter on an empty line** to submit.  
4. The character should speak and the face should move.

---

## Input format

```json
{"sentences": [
  {"text": "Hello there, how are you today?", "emotion": "happy", "intensity": 0.75},
  {"text": "That is really sad news.", "emotion": "sad", "intensity": 0.7}
]}
```

| Field | Description |
|--------|-------------|
| `text` | Words to speak |
| `emotion` | One of: `neutral`, `happy`, `sad`, `angry`, `surprised`, `disgusted`, `fearful`, `sarcastic`, `thinking` |
| `intensity` | Strength from `0.0` to `1.0` |

---

## Folder layout after setup

```
project/
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
  upcoming_features.txt
  face.blend
  models/
    README.md
    parler-tts-mini-v1/     ← you download this
    wav2arkit_cpu/          ← you download this
  temp/                     ← created automatically while running
```

---

## Optional: Rhubarb lip-sync

Default lip-sync is **wav2arkit**. To use Rhubarb instead, set in `orchestrator.py`:

```python
LIPSYNC_ENGINE = "rhubarb"
```

Then:

1. Download: https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest  
2. Place `rhubarb.exe` in the project root.  
3. Copy the `res` folder from the release next to `rhubarb.exe`.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Face does not move | Confirm the Blender receiver is running; check for a “CONNECTED” message; ensure nothing else is using port 9001 |
| Parler fails to load | Confirm `models/parler-tts-mini-v1/` is complete; reinstall parler-tts; match torch and torchaudio versions |
| Lip-sync fails | Confirm `models/wav2arkit_cpu/` has both `.onnx` and `.onnx.data`; install `onnxruntime` |
| No audio | Check speakers and that `sounddevice` works on your default device |
| Slow speech generation | Expected with Parler Mini on many laptops; a CUDA GPU helps a lot |
| Shape keys ignored | Key names should match ARKit style (`jawOpen`, `mouthSmileLeft`, etc.) |

---

## License

- Project source in this repository: use under the license you attach to the repo.  
- **Parler-TTS**, **wav2arkit**, Rhubarb, and other third-party tools: follow their licenses on Hugging Face / GitHub.  
- Do not commit API keys or large model weight files.

For planned work beyond this release, see **`upcoming_features.txt`**.
