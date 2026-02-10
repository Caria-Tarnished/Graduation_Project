# Streamlit UI Startup Script
# Purpose: Activate virtual environment and start Streamlit application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Financial Analysis System - Streamlit UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[1/4] Cleaning Python cache..." -ForegroundColor Yellow

# Clean Python cache files
Get-ChildItem -Path "app" -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "app" -Recurse -Filter "*.pyc" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Python cache cleaned" -ForegroundColor Green

Write-Host "[2/4] Checking virtual environment..." -ForegroundColor Yellow

# Check if virtual environment exists
$VenvPath = Join-Path $ProjectRoot ".venv"

if (-Not (Test-Path $VenvPath)) {
    Write-Host "Error: Virtual environment directory not found!" -ForegroundColor Red
    Write-Host "Expected path: $VenvPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please create virtual environment first:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor White
    Write-Host "  .venv\Scripts\activate" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "Virtual environment found: $VenvPath" -ForegroundColor Green

# Check Python executable
$PythonExe = Join-Path $VenvPath "python.exe"
$StreamlitExe = Join-Path $VenvPath "Scripts\streamlit.exe"

if (-Not (Test-Path $PythonExe)) {
    Write-Host "Error: Python executable not found!" -ForegroundColor Red
    Write-Host "Expected path: $PythonExe" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "[3/4] Preparing environment..." -ForegroundColor Yellow

# Add virtual environment Scripts directory to PATH
$env:PATH = (Join-Path $VenvPath "Scripts") + ";" + $env:PATH

Write-Host "Environment ready" -ForegroundColor Green

Write-Host "[4/4] Starting Streamlit application..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Tips:" -ForegroundColor Cyan
Write-Host "  - Application will open in browser automatically" -ForegroundColor White
Write-Host "  - Default address: http://localhost:8501" -ForegroundColor White
Write-Host "  - Press Ctrl+C to stop server" -ForegroundColor White
Write-Host ""

# Start Streamlit
$AppPath = Join-Path $ProjectRoot "app\hosts\streamlit_app\app.py"

if (-Not (Test-Path $AppPath)) {
    Write-Host "Error: Streamlit application file not found!" -ForegroundColor Red
    Write-Host "Expected path: $AppPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if Streamlit is installed
if (-Not (Test-Path $StreamlitExe)) {
    Write-Host "Error: Streamlit not installed!" -ForegroundColor Red
    Write-Host "Please run:" -ForegroundColor Red
    Write-Host "  .venv\Scripts\activate" -ForegroundColor White
    Write-Host "  pip install streamlit" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

try {
    # Use streamlit executable from virtual environment
    & $StreamlitExe run $AppPath
} catch {
    Write-Host ""
    Write-Host "Error: Failed to start Streamlit!" -ForegroundColor Red
    Write-Host "Error message: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible causes:" -ForegroundColor Yellow
    Write-Host "  1. Port 8501 is already in use" -ForegroundColor White
    Write-Host "  2. Dependencies not fully installed: pip install -r requirements.txt" -ForegroundColor White
    Write-Host "  3. .env file not configured correctly" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
