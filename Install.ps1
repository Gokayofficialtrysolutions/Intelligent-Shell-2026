# Intelligent Shell 2026 - Installation Script
# This script will set up the Intelligent Shell environment

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              INTELLIGENT SHELL 2026 INSTALLER                 ║" -ForegroundColor Cyan
Write-Host "║                  by Gokaytrysolutions                          ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

Write-Host "🔍 Checking system requirements..." -ForegroundColor Yellow

# Check PowerShell version
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Host "❌ PowerShell 5.0 or higher required. Current version: $($PSVersionTable.PSVersion)" -ForegroundColor Red
    exit 1
}
Write-Host "✅ PowerShell version: $($PSVersionTable.PSVersion)" -ForegroundColor Green

# Check if Git is available
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "✅ Git is available" -ForegroundColor Green
} else {
    Write-Host "⚠️  Git not found. Some features may be limited." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Installing Intelligent Shell 2026..." -ForegroundColor Cyan

# Create directories if needed
$shellPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$profileDir = Split-Path -Parent $PROFILE

if (!(Test-Path $profileDir)) {
    New-Item -Type Directory -Path $profileDir -Force | Out-Null
    Write-Host "✅ Created PowerShell profile directory" -ForegroundColor Green
}

# Install the main script
if (Test-Path "$shellPath\IntelligentShell.ps1") {
    Write-Host "✅ Intelligent Shell script found" -ForegroundColor Green
} else {
    Write-Host "❌ IntelligentShell.ps1 not found in current directory" -ForegroundColor Red
    exit 1
}

# Backup existing profile if it exists
if (Test-Path $PROFILE) {
    $backupPath = "$PROFILE.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $PROFILE $backupPath
    Write-Host "✅ Backed up existing profile to $backupPath" -ForegroundColor Green
}

# Create or update PowerShell profile
$profileContent = @"
# Intelligent Shell 2026 Auto-load
# Added on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
if (Test-Path '$shellPath\IntelligentShell.ps1') {
    . '$shellPath\IntelligentShell.ps1' -Start
} else {
    Write-Warning "Intelligent Shell 2026 not found at $shellPath"
}
"@

if (Test-Path $PROFILE) {
    # Check if already added
    $existingContent = Get-Content $PROFILE -Raw
    if ($existingContent -notlike "*Intelligent Shell 2026*") {
        Add-Content -Path $PROFILE -Value "`n$profileContent"
        Write-Host "✅ Added Intelligent Shell to existing profile" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  Intelligent Shell already configured in profile" -ForegroundColor Yellow
    }
} else {
    # Create new profile
    New-Item -Type File -Path $PROFILE -Force | Out-Null
    Set-Content -Path $PROFILE -Value $profileContent
    Write-Host "✅ Created new PowerShell profile with Intelligent Shell" -ForegroundColor Green
}

# Set execution policy if needed
try {
    $executionPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($executionPolicy -eq "Restricted") {
        if ($isAdmin) {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine -Force
            Write-Host "✅ Set execution policy to RemoteSigned (LocalMachine)" -ForegroundColor Green
        } else {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Write-Host "✅ Set execution policy to RemoteSigned (CurrentUser)" -ForegroundColor Green
        }
    } else {
        Write-Host "✅ Execution policy is already suitable: $executionPolicy" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Could not modify execution policy. You may need to run as Administrator." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Intelligent Shell 2026 installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Close and reopen PowerShell" -ForegroundColor White
Write-Host "   2. The Intelligent Shell will start automatically" -ForegroundColor White
Write-Host "   3. Type 'ihelp' to see available commands" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Manual activation (if needed):" -ForegroundColor Cyan
Write-Host "   . '$shellPath\IntelligentShell.ps1' -Start" -ForegroundColor White
Write-Host ""

# Offer to start immediately
$response = Read-Host "Would you like to start Intelligent Shell now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq 'yes') {
    Write-Host ""
    . "$shellPath\IntelligentShell.ps1" -Start
}
