# Development Setup Script for Baseball Analytics Monorepo
#
# This script installs all packages in editable mode for development

Write-Host "=" * 70
Write-Host "Baseball Analytics Monorepo - Development Setup"
Write-Host "=" * 70
Write-Host ""

# Check if we're in the right directory
$expectedDir = "baseball_analytics"
$currentDir = Split-Path -Leaf (Get-Location)

if ($currentDir -ne $expectedDir) {
    Write-Host "Error: This script must be run from the baseball_analytics root directory" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}

# Install packages in editable mode
Write-Host ""
Write-Host "Installing packages in editable mode..." -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Installing matchup_machine..." -ForegroundColor Yellow
pip install -e packages/matchup_machine

Write-Host ""
Write-Host "2. Installing fantasy_mlb_ai..." -ForegroundColor Yellow
pip install -e packages/fantasy_mlb_ai

Write-Host ""
Write-Host "3. Installing diamond_mind..." -ForegroundColor Yellow
pip install -e packages/diamond_mind

Write-Host ""
Write-Host "=" * 70
Write-Host "✓ Development setup complete!" -ForegroundColor Green
Write-Host "=" * 70
Write-Host ""
Write-Host "You can now import packages:"
Write-Host "  from matchup_machine import load_artifacts" -ForegroundColor Cyan
Write-Host "  from fantasy_mlb_ai import MLProjectionEngine" -ForegroundColor Cyan
Write-Host "  from diamond_mind.shared import settings" -ForegroundColor Cyan
Write-Host ""
