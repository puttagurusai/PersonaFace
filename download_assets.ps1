# download_assets.ps1 — fetch large weights into models/ (run from git_upload3 root)
# Requires: pip install huggingface_hub ; huggingface-cli login (for gated A2E)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== TalkFace Brain — download assets ===" -ForegroundColor Cyan
Write-Host "Root: $Root"

function Ensure-Dir($p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

Ensure-Dir "models\parler-tts-mini-v1"
Ensure-Dir "models\wav2arkit_cpu"
Ensure-Dir "models\audio2emotion_v2.2"
Ensure-Dir "models\brain\hubert-base-ls960"

# --- Parler Mini ---
if (-not (Test-Path "models\parler-tts-mini-v1\model.safetensors") -and
    -not (Test-Path "models\parler-tts-mini-v1\pytorch_model.bin")) {
    Write-Host "`n[1/4] Parler-TTS Mini (~3.5 GB)..." -ForegroundColor Yellow
    huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models/parler-tts-mini-v1
} else {
    Write-Host "[1/4] Parler Mini already present — skip"
}

# --- wav2arkit ---
if (-not (Test-Path "models\wav2arkit_cpu\wav2arkit_cpu.onnx")) {
    Write-Host "`n[2/4] wav2arkit_cpu..." -ForegroundColor Yellow
    huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models/wav2arkit_cpu
} else {
    Write-Host "[2/4] wav2arkit already present — skip"
}

# --- HuBERT ---
if (-not (Test-Path "models\brain\hubert-base-ls960\config.json")) {
    Write-Host "`n[3/4] HuBERT base..." -ForegroundColor Yellow
    huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960
} else {
    Write-Host "[3/4] HuBERT already present — skip"
}

# --- Audio2Emotion (gated) ---
if (-not (Test-Path "models\audio2emotion_v2.2\network.onnx")) {
    Write-Host "`n[4/4] Audio2Emotion-v2.2 (~1.27 GB, gated)..." -ForegroundColor Yellow
    Write-Host "  Accept license: https://huggingface.co/nvidia/Audio2Emotion-v2.2"
    Write-Host "  Then: huggingface-cli login"
    python audio2emotion.py --download
} else {
    Write-Host "[4/4] Audio2Emotion already present — skip"
}

Write-Host "`n=== Brain .pt modules (manual) ===" -ForegroundColor Cyan
Write-Host "Place these yourself (from training export):"
Write-Host "  models/brain/shared_encoder.pt"
Write-Host "  models/brain/face_head.pt"
Write-Host "  models/brain/character_adapter.pt"
Write-Host "Or: python export_brain_modules.py path\to\brain_latest.pt"

Write-Host "`nDone. Run: python check_setup.py" -ForegroundColor Green
