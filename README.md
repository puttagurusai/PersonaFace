# TalkFace

Real-time talking-head control for **Blender**.

Drive a Faceit / ARKit-rigged face with:

1. **Speech mode** — text → voice (Kokoro TTS) → lip-sync (Rhubarb) → emotion blendshapes  
2. **Live mode** — webcam or video → MediaPipe → blendshapes  

Both send data over UDP to Blender (`127.0.0.1:9001`). The face eases back to neutral smoothly at the end of each spoken line.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** | System Python (not Blender’s built-in Python) |
| **Blender** | Scene with ARKit-style shape keys (download form here [`face.blend`](https://dragonboots.gumroad.com/l/metahumanhead) |
| **Windows** | Primary path (`rhubarb.exe`). Other OS: use the matching Rhubarb binary named `rhubarb` |
| **Microphone / speakers** | Speakers for TTS playback; webcam only for live capture |

---

## Project files

| File | What it does |
|------|----------------|
| `orchestrator.py` | Main app: TTS, lip-sync, emotions, timed mouth shapes |
| `blender_receiver.py` | **Run this inside Blender** — listens for animation data |
| `emotion_map.py` | Emotion → upper-face blendshape presets |
| `mediapipe_face_capture.py` | Live webcam / video face capture |
| `blender_face_receiver.py` | Older MediaPipe-only receiver (optional / reference) |
| `check_setup.py` | Checks that models and packages are installed |
| `download_assets.ps1` | Downloads large model files (not in git) |
| `face.blend` | Example Blender face scene |
| `requirements.txt` | Python dependencies |

---

## Setup (first time)

### 1. Install Python packages

```powershell
pip install -r requirements.txt
```

### 2. Download large assets (not in this repo)

From the project folder:

```powershell
.\download_assets.ps1
```

This downloads:

- `kokoro-v1.0.onnx` (~310 MB) — TTS model  
- `voices-v1.0.bin` — voice data  
- `face_landmarker.task` — MediaPipe face model (live mode)

**Manual links if the script fails:**

- Kokoro: [model-files-v1.0](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)  
- Face landmarker:  
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`

### 3. Install Rhubarb Lip Sync

1. Download the latest release:  
   [rhubarb-lip-sync releases](https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest)
2. Put **`rhubarb.exe`** in the project root (same folder as `orchestrator.py`).
3. Copy the **`res`** folder from the zip next to `rhubarb.exe`.  
   Phonetic lip-sync will fail without `res/`.

Your folder should look like:

```
TalkFace/
  orchestrator.py
  blender_receiver.py
  ...
  rhubarb.exe
  res/
    sphinx/
      ...
  kokoro-v1.0.onnx
  voices-v1.0.bin
  face_landmarker.task    # only needed for live capture
```

### 4. Check setup

```powershell
python check_setup.py
```

Fix anything marked missing before continuing.

> **Note:** An Anthropic API key is **not** required for the current mode. You paste JSON sentences; Claude is optional for later.

---

## How to run — Speech mode (main)

This is the normal way to use the project: character speaks with lip-sync and emotion.

### Step A — Start Blender receiver

1. Open **`face.blend`** in Blender (or your own Faceit / ARKit mesh scene).
2. Open **`blender_receiver.py`** in Blender’s Text Editor.
3. Click **Run Script**.
4. In Blender’s **Python Console**, run:

```python
bpy.ops.face.stream_receiver()
```

You should see shape-key / listener messages in the Blender system console. Leave Blender running.

To stop later:

```python
bpy.ops.face.stream_stop()
```

### Step B — Start the orchestrator

In a terminal, from the project folder:

```powershell
python orchestrator.py
```

1. Press **Enter** only after the Blender receiver is already running.  
2. Paste a JSON block, then press **Enter on an empty line** to submit.

**Example input:**

```json
{"sentences": [
  {"text": "Hello there, how are you today?", "emotion": "happy", "intensity": 0.75},
  {"text": "I'm not sure I like where this is going.", "emotion": "fearful", "intensity": 0.7}
]}
```

3. For each sentence the pipeline will:
   - generate speech (Kokoro)
   - run lip-sync (Rhubarb)
   - play audio
   - stream mouth shapes + emotion to Blender
   - ease the face back to neutral when the line ends

Type `quit` on its own line to exit.

### Supported emotions

`neutral` · `happy` · `sad` · `angry` · `surprised` · `disgusted` · `fearful` · `sarcastic` · `thinking`

`intensity` is a float from `0.0` to `1.0`.

---

## How to run — Live capture mode (optional)

Mirror a real face (webcam or video) onto the Blender mesh.

### Step A — Same Blender receiver

Use `blender_receiver.py` and:

```python
bpy.ops.face.stream_receiver()
```

### Step B — Start capture

```powershell
python mediapipe_face_capture.py
```

A preview window opens. The Blender face should follow expressions.

- Press **`q`** in the preview to quit.  
- Or stop the receiver with `bpy.ops.face.stream_stop()`.

**Tips:** even lighting, face the camera, plain background.

---

## Quick reference — what to run

| Goal | Run this |
|------|----------|
| Character talks from text | 1) Blender: `blender_receiver.py` + `bpy.ops.face.stream_receiver()`  
2) Terminal: `python orchestrator.py` |
| Live webcam face | 1) Same Blender receiver  
2) Terminal: `python mediapipe_face_capture.py` |
| Verify install | `python check_setup.py` |
| Download models | `.\download_assets.ps1` |
| Stop Blender listener | `bpy.ops.face.stream_stop()` |

**Always start the Blender receiver first**, then the Python sender.

---

## Order of operations (speech mode)

```
1. Open face.blend in Blender
2. Run blender_receiver.py  →  bpy.ops.face.stream_receiver()
3. python orchestrator.py
4. Paste JSON  →  blank line
5. Watch the face speak in Blender
```

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Face doesn’t move | Confirm receiver is running; check Blender console for `[face_receiver]` logs |
| Shape keys ignored | Names must match ARKit style (`jawOpen`, `mouthSmileLeft`, …). Check the list printed when the receiver starts |
| Rhubarb errors | `rhubarb.exe` in project root; `res/` folder next to it |
| No sound | Speakers on; `sounddevice` installed; Windows audio device working |
| Kokoro fails to load | `kokoro-v1.0.onnx` and `voices-v1.0.bin` in project root |
| Nothing after paste | JSON must include `"sentences"`; submit with a **blank line** after the paste |
| Sudden freeze between lines | Receiver should stay on; leave it running for the whole session |

---

## How it works (short)

```
Text JSON
   → Kokoro TTS  (WAV)
   → Rhubarb     (mouth cues A–H / X)
   → UDP packets:
        type=emotion  → upper face (brows, eyes, smile…)
        type=viseme   → mouth / jaw
        type=rest_pose (smooth) → ease to neutral
   → blender_receiver.py applies shape keys on the mesh
```

Live mode skips TTS/Rhubarb and sends full-face MediaPipe blendshapes instead.

---

## License notes

- Project code in this repo is yours to use under whatever license you choose.  
- **Kokoro**, **Rhubarb**, and **MediaPipe** are third-party; follow their licenses.  
- Do not commit API keys or large model binaries to git — download them locally after clone.
