# download_assets.ps1
# Downloads large assets that are NOT committed to git (see .gitignore / README).
# Run with:  .\download_assets.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Downloading project assets (not in git) ===" -ForegroundColor Cyan

$files = @(
    @{
        Url  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        Dest = "kokoro-v1.0.onnx"
        Size = "~310 MB"
    },
    @{
        Url  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        Dest = "voices-v1.0.bin"
        Size = "~27 MB"
    },
    @{
        Url  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        Dest = "face_landmarker.task"
        Size = "~4 MB"
    }
)

foreach ($file in $files) {
    if (Test-Path $file.Dest) {
        Write-Host "Already exists: $($file.Dest)" -ForegroundColor Yellow
        continue
    }

    Write-Host "Downloading $($file.Dest) ($($file.Size)) ..." -ForegroundColor White
    try {
        # Use curl.exe (built into Windows) with -L for redirect + progress
        & curl.exe -L --progress-bar -o $file.Dest $file.Url
        Write-Host "Downloaded $($file.Dest)" -ForegroundColor Green
    } catch {
        Write-Host "Failed to download $($file.Dest)" -ForegroundColor Red
        Write-Host "Please download manually from:" -ForegroundColor Yellow
        Write-Host $file.Url
    }
}

Write-Host ""
Write-Host "=== Rhubarb (manual step) ===" -ForegroundColor Cyan
Write-Host "1. Go to: https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest"
Write-Host "2. Download the Windows zip"
Write-Host "3. Extract rhubarb.exe into this folder"
Write-Host "4. Copy the 'res' folder from the zip next to rhubarb.exe (needed for phonetic mode)"

Write-Host ""
Write-Host "After assets are present:" -ForegroundColor Green
Write-Host "  pip install -r requirements.txt"
Write-Host "  python check_setup.py"
Write-Host "  # In Blender: run blender_receiver.py then bpy.ops.face.stream_receiver()"
Write-Host "  python orchestrator.py"
Write-Host ""
