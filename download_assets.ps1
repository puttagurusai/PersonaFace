# download_assets.ps1 — orchestrator pipeline models → models/
# Run:  .\download_assets.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host "=== TalkFace orchestrator assets → models/ ===" -ForegroundColor Cyan

function Download-HF($Repo, $Dest) {
    if ((Test-Path $Dest) -and (Get-ChildItem $Dest -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count -gt 3) {
        Write-Host "Already has files: $Dest" -ForegroundColor Yellow
        return
    }
    Write-Host "Downloading $Repo → $Dest ..." -ForegroundColor White
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    huggingface-cli download $Repo --local-dir $Dest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CLI failed; try:" -ForegroundColor Red
        Write-Host "  python -c `"from huggingface_hub import snapshot_download; snapshot_download(repo_id='$Repo', local_dir=r'$Dest', local_dir_use_symlinks=False)`""
    } else {
        Write-Host "OK: $Dest" -ForegroundColor Green
    }
}

Download-HF "parler-tts/parler-tts-mini-v1" "models\parler-tts-mini-v1"
Download-HF "myned-ai/wav2arkit_cpu" "models\wav2arkit_cpu"

Write-Host ""
Write-Host "Next:" -ForegroundColor Green
Write-Host "  pip install -r requirements.txt"
Write-Host "  pip install git+https://github.com/huggingface/parler-tts.git"
Write-Host "  python check_setup.py"
Write-Host "  # Blender: blender_receiver.py → bpy.ops.face.stream_receiver()"
Write-Host "  python orchestrator.py"
