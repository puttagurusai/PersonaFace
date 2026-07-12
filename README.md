# ProjFace — Live Face Capture + AI Talking Head

Drive a Faceit / ARKit-rigged character in **Blender** from:

1. **Live capture** — webcam or video → MediaPipe blendshapes → UDP → Blender  
2. **Conversational pipeline** — text (JSON) → Kokoro TTS + Rhubarb lip-sync + emotions → UDP → Blender  

Both modes use the same Blender receiver on `127.0.0.1:9001`.

---

## Project layout (source)

| File | Role |
|------|------|
| `orchestrator.py` | Main pipeline: TTS, Rhubarb, timed visemes + emotions over UDP |
| `emotion_map.py` | Upper-face emotion → ARKit blendshape presets |
| `blender_receiver.py` | **Use this in Blender** (emotion + viseme + MediaPipe) |
| `blender_face_receiver.py` | Legacy MediaPipe-only receiver (reference) |
| `mediapipe_face_capture.py` | Webcam / video → blendshapes over UDP |
| `check_setup.py` | Verifies models, packages, and optional API key |
| `download_assets.ps1` | Downloads Kokoro model files on Windows |
| `face.blend` | Example Blender scene with face mesh |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps large binaries and temp files out of git |

---

## What to put in Git

### Commit these

```
.gitignore
README.md
requirements.txt
orchestrator.py
emotion_map.py
blender_receiver.py
blender_face_receiver.py
mediapipe_face_capture.py
check_setup.py
download_assets.ps1
face.blend                 # optional but useful (~10 MB)
```

### Do **not** commit (ignored by `.gitignore`)

| Path | Why |
|------|-----|
| `kokoro-v1.0.onnx` (~310 MB) | TTS model — over GitHub’s 100 MB limit |
| `voices-v1.0.bin` (~28 MB) | Voice embeddings — download with script |
| `face_landmarker.task` (~4 MB) | MediaPipe model — download once |
| `rhubarb.exe` | Third-party binary — download from releases |
| `Rhubarb-Lip-Sync-*/` and `*.zip` | Full vendor tree + zip (~80–150 MB) |
| `res/` | Rhubarb sphinx resources — copy from Rhubarb release |
| `temp/` | Generated WAV + Rhubarb JSON per sentence |
| `__pycache__/` | Python bytecode |
| `*.blend1` | Blender autosave backups |
| `testing.mp4` | Local test video (optional sample) |
| `mcps/` | Local tooling, not part of the app |

After clone, each developer downloads the large assets (see below). Do not force-add ignored model files.

---

## Requirements

- **Python 3.10+** (system Python — not Blender’s)
- **Blender** with a mesh that has ARKit-style shape keys (e.g. Faceit)
- **Windows** is the primary path for Rhubarb (`rhubarb.exe`); Linux/macOS can use the matching Rhubarb binary named `rhubarb`

### Python packages

```powershell
pip install -r requirements.txt
```

Packages:

- **Capture:** `mediapipe`, `opencv-python`
- **Orchestrator:** `kokoro-onnx`, `sounddevice`, `soundfile`, `numpy`, `anthropic`

---

## One-time asset setup

Large files stay **out of git**. Place them in the project root after clone.

### 1. Kokoro TTS

```powershell
.\download_assets.ps1
```

Or download manually from  
[kokoro-onnx model-files-v1.0](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0):

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

### 2. Rhubarb Lip Sync

1. Download the latest release:  
   [rhubarb-lip-sync releases](https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest)
2. Extract **`rhubarb.exe`** into the project root.
3. Copy the **`res`** folder from the zip next to `rhubarb.exe`  
   (required for `-r phonetic`; without it Rhubarb fails resource lookup).

Layout after setup:

```
projface_v1/
  rhubarb.exe
  res/
    sphinx/
      ...
  kokoro-v1.0.onnx
  voices-v1.0.bin
  ...
```

### 3. MediaPipe face landmarker (live capture only)

Download (~4 MB) into the project root:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 4. Optional: Anthropic API key

Current orchestrator mode accepts **pasted JSON** (no live LLM required).  
When you enable Claude later:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 5. Verify

```powershell
python check_setup.py
```

---

## Mode A — Live MediaPipe capture

1. Install deps: `pip install -r requirements.txt`
2. Place `face_landmarker.task` in the project root.
3. In Blender: open **`blender_receiver.py`** → Run Script →  
   ```python
   bpy.ops.face.stream_receiver()
   ```
4. On system Python:
   ```powershell
   python mediapipe_face_capture.py
   ```
5. Stop: `bpy.ops.face.stream_stop()` in Blender, or `q` in the capture window.

Capture tips: even lighting, face the camera, plain background.

---

## Mode B — Orchestrator (TTS + lip-sync + emotion)

1. Finish [One-time asset setup](#one-time-asset-setup) (Kokoro + Rhubarb + `res/`).
2. In Blender: open **`blender_receiver.py`** → Run Script →  
   ```python
   bpy.ops.face.stream_receiver()
   ```
3. Run:
   ```powershell
   python orchestrator.py
   ```
4. Paste JSON, then a blank line to submit. Example:

```json
{"sentences": [
  {"text": "Hello there, how are you today?", "emotion": "happy", "intensity": 0.75},
  {"text": "I'm not sure I like where this is going.", "emotion": "fearful", "intensity": 0.7}
]}
```

Supported emotions:  
`neutral`, `happy`, `sad`, `angry`, `surprised`, `disgusted`, `fearful`, `sarcastic`, `thinking`.

Type `quit` on its own line to exit.

At the end of each sentence the face **eases** back to neutral (mouth settle + soft rest pose), not a hard snap.

---

## UDP protocol (for integrators)

Receiver listens on **`127.0.0.1:9001`**.

| `type` | Effect |
|--------|--------|
| `emotion` | Upper face (brows, eyes, smile/frown, …) |
| `viseme` | Mouth / jaw only |
| `rest_pose` | Full face rest; `"smooth": true` glides, omit/false snaps |
| `head` | Optional head pitch/yaw/roll |
| *(no type / legacy)* | Full face (MediaPipe) |

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Nothing moves | Blender console `[face_receiver]` logs; shape key names vs ARKit (`jawOpen`, `mouthSmileLeft`, …) |
| Rhubarb fails | `res/` next to `rhubarb.exe`; run phonetic mode only with that tree present |
| No audio | Working speakers; `sounddevice` installed; not muted |
| Kokoro missing | `kokoro-v1.0.onnx` + `voices-v1.0.bin` in project root |
| Laggy face | Lower smoothing in `blender_receiver.py`; keep sender and Blender on the same machine |
| Jerk at end of speech | Re-run updated `blender_receiver.py` + `orchestrator.py` (smooth rest path) |

---

## License notes

- **Your code** in this repo is yours to license as you choose.
- **Kokoro**, **Rhubarb**, **MediaPipe**, and voice/model files are third-party — follow their licenses when redistributing.
- Do not commit proprietary API keys. Use environment variables only.
