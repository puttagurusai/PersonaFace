# Brain weights (`models/brain/`)

## Included in this package (small)

| File | Role |
|------|------|
| `emotion_stats.json` | Fallback 26-D averages if Audio2Emotion is missing |
| `emotion_26d_layout.json` | Maps A2E 6-class → 26-D layout |

## You must add (large / trained)

| File | Role |
|------|------|
| `shared_encoder.pt` | HuBERT path + fusion encoder |
| `face_head.pt` | Lower/upper face heads + ARKit indices |
| `character_adapter.pt` or `adapter.pt` | Residual character adapter |
| `hubert-base-ls960/` | Local HuBERT (config + weights) |

Export from a full training checkpoint:

```powershell
python export_brain_modules.py path\to\brain_latest.pt
```

HuBERT:

```powershell
huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960
```
