# models/ — download here (not committed to git)

## Required for orchestrator.py

### 1. Parler-TTS Mini v1 (~3.5 GB)

- **Link:** https://huggingface.co/parler-tts/parler-tts-mini-v1  
- **Save to:** `models/parler-tts-mini-v1/`

```powershell
huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models\parler-tts-mini-v1
```

Need at least: `config.json`, `model.safetensors`.

### 2. wav2arkit CPU (~385 MB)

- **Link:** https://huggingface.co/myned-ai/wav2arkit_cpu  
- **Save to:** `models/wav2arkit_cpu/`

```powershell
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models\wav2arkit_cpu
```

Need: `wav2arkit_cpu.onnx`, `wav2arkit_cpu.onnx.data`, `config.json`.

## Optional

| Asset | Link | Path |
|-------|------|------|
| Rhubarb lip-sync | https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest | project root `rhubarb.exe` + `res/` |

Only if you set `LIPSYNC_ENGINE = "rhubarb"` in `orchestrator.py`.
