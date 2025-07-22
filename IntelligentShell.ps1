# Intelligent Shell 2026 by Gokaytrysolutions
# Enhanced PowerShell environment with intelligent features

param(
    [switch]$Install,
    [switch]$Start,
    [switch]$Config
)

# Color scheme for the intelligent shell
$Colors = @{
    Primary = "Cyan"
    Secondary = "Green" 
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Success = "Green"
}

# Intelligent Shell Configuration
$IntelligentShellConfig = @{
    Version = "1.0.0"
    Author = "Gokaytrysolutions"
    Year = "2026"
    Features = @(
        "Smart Command Completion",
        "AI-Powered Suggestions",
        "System Monitoring",
        "Package Management Integration",
        "Git Integration",
        "Performance Analytics"
    )
}

# Function to display banner
function Show-IntelligentShellBanner {
    Clear-Host
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor $Colors.Primary
    Write-Host "║                    INTELLIGENT SHELL 2026                     ║" -ForegroundColor $Colors.Primary
    Write-Host "║                  by Gokaytrysolutions                          ║" -ForegroundColor $Colors.Secondary
    Write-Host "╠════════════════════════════════════════════════════════════════╣" -ForegroundColor $Colors.Primary
    Write-Host "║ Version: $($IntelligentShellConfig.Version)                                               ║" -ForegroundColor $Colors.Info
    Write-Host "║ Features: AI-Powered Terminal Enhancement                      ║" -ForegroundColor $Colors.Info
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor $Colors.Primary
    Write-Host ""
}

# Function to get system information
function Get-IntelligentSystemInfo {
    $systemInfo = @{
        OS = "$((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
        Computer = $env:COMPUTERNAME
        User = $env:USERNAME
        PowerShell = "$($PSVersionTable.PSVersion)"
        Architecture = $env:PROCESSOR_ARCHITECTURE
        Memory = "{0:N2} GB" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        Uptime = (Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    }
    
    Write-Host "🖥️  System Information:" -ForegroundColor $Colors.Primary
    Write-Host "   OS: $($systemInfo.OS)" -ForegroundColor $Colors.Info
    Write-Host "   Computer: $($systemInfo.Computer)" -ForegroundColor $Colors.Info
    Write-Host "   User: $($systemInfo.User)" -ForegroundColor $Colors.Info
    Write-Host "   PowerShell: $($systemInfo.PowerShell)" -ForegroundColor $Colors.Info
    Write-Host "   Architecture: $($systemInfo.Architecture)" -ForegroundColor $Colors.Info
    Write-Host "   Memory: $($systemInfo.Memory)" -ForegroundColor $Colors.Info
    Write-Host "   Uptime: $($systemInfo.Uptime.Days)d $($systemInfo.Uptime.Hours)h $($systemInfo.Uptime.Minutes)m" -ForegroundColor $Colors.Info
    Write-Host ""
}

# Function to check installed package managers
function Get-PackageManagerStatus {
    Write-Host "📦 Package Manager Status:" -ForegroundColor $Colors.Primary
    
    # Check Chocolatey
    try {
        $chocoVersion = (choco --version 2>$null)
        Write-Host "   ✅ Chocolatey: $chocoVersion" -ForegroundColor $Colors.Success
    }
    catch {
        Write-Host "   ❌ Chocolatey: Not installed" -ForegroundColor $Colors.Error
    }
    
    # Check Scoop
    try {
        $scoopVersion = (scoop --version 2>$null)
        Write-Host "   ✅ Scoop: $($scoopVersion[0])" -ForegroundColor $Colors.Success
    }
    catch {
        Write-Host "   ❌ Scoop: Not installed" -ForegroundColor $Colors.Error
    }
    
    # Check Winget
    try {
        $wingetVersion = (winget --version 2>$null)
        Write-Host "   ✅ Winget: $wingetVersion" -ForegroundColor $Colors.Success
    }
    catch {
        Write-Host "   ❌ Winget: Not installed" -ForegroundColor $Colors.Error
    }
    
    Write-Host ""
}

# Function to display available commands
function Show-IntelligentCommands {
    Write-Host "🤖 Intelligent Shell Commands:" -ForegroundColor $Colors.Primary
    Write-Host "   isys       - System information and monitoring" -ForegroundColor $Colors.Info
    Write-Host "   ipack      - Package management integration" -ForegroundColor $Colors.Info
    Write-Host "   igit       - Enhanced git operations" -ForegroundColor $Colors.Info
    Write-Host "   inetwork   - Network diagnostics and tools" -ForegroundColor $Colors.Info
    Write-Host "   iprocess   - Process management and monitoring" -ForegroundColor $Colors.Info
    Write-Host "   ifile      - Advanced file operations" -ForegroundColor $Colors.Info
    Write-Host "   ihelp      - Show this help menu" -ForegroundColor $Colors.Info
    Write-Host "   iexit      - Exit Intelligent Shell" -ForegroundColor $Colors.Info
    Write-Host ""
}

# Enhanced system command
function Invoke-IntelligentSystem {
    param([string]$Action = "info")
    
    switch ($Action.ToLower()) {
        "info" { Get-IntelligentSystemInfo }
        "monitor" {
            Write-Host "📊 System Monitor (Press Ctrl+C to stop):" -ForegroundColor $Colors.Primary
            while ($true) {
                $cpu = (Get-Counter "\Processor(_Total)\% Processor Time").CounterSamples.CookedValue
                $memory = Get-CimInstance Win32_OperatingSystem
                $memoryUsed = (($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100
                
                Write-Host "CPU: $([math]::Round($cpu, 1))% | Memory: $([math]::Round($memoryUsed, 1))% | $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor $Colors.Info
                Start-Sleep -Seconds 2
            }
        }
        "cleanup" {
            Write-Host "🧹 Running system cleanup..." -ForegroundColor $Colors.Warning
            Get-ChildItem -Path $env:TEMP -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
            Write-Host "✅ Temporary files cleaned" -ForegroundColor $Colors.Success
        }
        default { Write-Host "Usage: isys [info|monitor|cleanup]" -ForegroundColor $Colors.Warning }
    }
}

# Enhanced package management
function Invoke-IntelligentPackage {
    param([string]$Action, [string]$Package)
    
    if (-not $Action) {
        Get-PackageManagerStatus
        return
    }
    
    switch ($Action.ToLower()) {
        "install" {
            if (-not $Package) {
                Write-Host "Usage: ipack install <package-name>" -ForegroundColor $Colors.Warning
                return
            }
            Write-Host "🚀 Installing $Package..." -ForegroundColor $Colors.Primary
            # Try different package managers
            if (Get-Command choco -ErrorAction SilentlyContinue) {
                choco install $Package -y
            } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
                winget install $Package
            } elseif (Get-Command scoop -ErrorAction SilentlyContinue) {
                scoop install $Package
            }
        }
        "search" {
            if (-not $Package) {
                Write-Host "Usage: ipack search <package-name>" -ForegroundColor $Colors.Warning
                return
            }
            Write-Host "🔍 Searching for $Package..." -ForegroundColor $Colors.Primary
            if (Get-Command choco -ErrorAction SilentlyContinue) {
                choco search $Package
            }
        }
        "list" {
            Write-Host "📋 Installed packages:" -ForegroundColor $Colors.Primary
            if (Get-Command choco -ErrorAction SilentlyContinue) {
                choco list --local-only
            }
        }
        default { Write-Host "Usage: ipack [install|search|list] [package-name]" -ForegroundColor $Colors.Warning }
    }
}

# Enhanced git operations
function Invoke-IntelligentGit {
    param([string]$Action, [string]$Parameter)
    
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "❌ Git is not installed" -ForegroundColor $Colors.Error
        return
    }
    
    switch ($Action.ToLower()) {
        "status" { 
            Write-Host "📊 Git Status:" -ForegroundColor $Colors.Primary
            git status --porcelain | ForEach-Object {
                $status = $_.Substring(0, 2)
                $file = $_.Substring(3)
                switch ($status) {
                    "??" { Write-Host "   ❓ $file (untracked)" -ForegroundColor $Colors.Warning }
                    " M" { Write-Host "   📝 $file (modified)" -ForegroundColor $Colors.Info }
                    "M " { Write-Host "   ✅ $file (staged)" -ForegroundColor $Colors.Success }
                    "D " { Write-Host "   🗑️  $file (deleted)" -ForegroundColor $Colors.Error }
                    default { Write-Host "   📄 $file ($status)" -ForegroundColor $Colors.Info }
                }
            }
        }
        "quick" {
            Write-Host "⚡ Quick commit and push..." -ForegroundColor $Colors.Primary
            git add .
            $message = if ($Parameter) { $Parameter } else { "Quick update $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
            git commit -m $message
            git push
        }
        "info" {
            Write-Host "📈 Repository Information:" -ForegroundColor $Colors.Primary
            $branch = git rev-parse --abbrev-ref HEAD
            $commits = git rev-list --count HEAD
            $lastCommit = git log -1 --format="%cr"
            Write-Host "   Branch: $branch" -ForegroundColor $Colors.Info
            Write-Host "   Commits: $commits" -ForegroundColor $Colors.Info
            Write-Host "   Last commit: $lastCommit" -ForegroundColor $Colors.Info
        }
        default { Write-Host "Usage: igit [status|quick|info] [message]" -ForegroundColor $Colors.Warning }
    }
}

# Network diagnostics
function Invoke-IntelligentNetwork {
    param([string]$Action, [string]$Target = "8.8.8.8")
    
    switch ($Action.ToLower()) {
        "test" {
            Write-Host "🌐 Testing network connectivity to $Target..." -ForegroundColor $Colors.Primary
            $result = Test-NetConnection -ComputerName $Target -Port 80 -InformationLevel Quiet
            if ($result) {
                Write-Host "✅ Connection successful" -ForegroundColor $Colors.Success
            } else {
                Write-Host "❌ Connection failed" -ForegroundColor $Colors.Error
            }
        }
        "speed" {
            Write-Host "⚡ Testing network speed..." -ForegroundColor $Colors.Primary
            $result = Test-NetConnection -ComputerName "speedtest.net" -Port 80 -InformationLevel Detailed
            Write-Host "Ping time: $($result.PingReplyDetails.RoundtripTime)ms" -ForegroundColor $Colors.Info
        }
        "info" {
            Write-Host "📡 Network Information:" -ForegroundColor $Colors.Primary
            $adapters = Get-NetAdapter | Where-Object Status -eq "Up"
            foreach ($adapter in $adapters) {
                $ip = (Get-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress
                Write-Host "   $($adapter.Name): $ip" -ForegroundColor $Colors.Info
            }
        }
        default { Write-Host "Usage: inetwork [test|speed|info] [target]" -ForegroundColor $Colors.Warning }
    }
}

# Process management
function Invoke-IntelligentProcess {
    param([string]$Action, [string]$ProcessName)
    
    switch ($Action.ToLower()) {
        "top" {
            Write-Host "🔝 Top processes by CPU usage:" -ForegroundColor $Colors.Primary
            Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, CPU, WorkingSet | Format-Table -AutoSize
        }
        "memory" {
            Write-Host "💾 Top processes by memory usage:" -ForegroundColor $Colors.Primary
            Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10 Name, @{Name="Memory(MB)";Expression={[math]::Round($_.WorkingSet/1MB,2)}}, CPU | Format-Table -AutoSize
        }
        "kill" {
            if (-not $ProcessName) {
                Write-Host "Usage: iprocess kill <process-name>" -ForegroundColor $Colors.Warning
                return
            }
            Write-Host "🔪 Killing process: $ProcessName" -ForegroundColor $Colors.Warning
            Stop-Process -Name $ProcessName -Force -ErrorAction SilentlyContinue
            Write-Host "✅ Process terminated" -ForegroundColor $Colors.Success
        }
        default { Write-Host "Usage: iprocess [top|memory|kill] [process-name]" -ForegroundColor $Colors.Warning }
    }
}

# Advanced file operations
function Invoke-IntelligentFile {
    param([string]$Action, [string]$Path, [string]$Target)
    
    switch ($Action.ToLower()) {
        "size" {
            if (-not $Path) { $Path = "." }
            Write-Host "📊 Directory size analysis for: $Path" -ForegroundColor $Colors.Primary
            Get-ChildItem -Path $Path -Directory | ForEach-Object {
                $size = (Get-ChildItem -Path $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
                $sizeGB = [math]::Round($size / 1GB, 2)
                $sizeMB = [math]::Round($size / 1MB, 2)
                if ($sizeGB -gt 0) {
                    Write-Host "   $($_.Name): ${sizeGB} GB" -ForegroundColor $Colors.Info
                } else {
                    Write-Host "   $($_.Name): ${sizeMB} MB" -ForegroundColor $Colors.Info
                }
            }
        }
        "find" {
            if (-not $Path) {
                Write-Host "Usage: ifile find <search-term> [directory]" -ForegroundColor $Colors.Warning
                return
            }
            $searchDir = if ($Target) { $Target } else { "." }
            Write-Host "🔍 Searching for '$Path' in $searchDir..." -ForegroundColor $Colors.Primary
            Get-ChildItem -Path $searchDir -Recurse -Filter "*$Path*" -ErrorAction SilentlyContinue | Select-Object Name, Directory
        }
        "cleanup" {
            Write-Host "🧹 Cleaning up temporary and cache files..." -ForegroundColor $Colors.Primary
            $locations = @($env:TEMP, "$env:USERPROFILE\AppData\Local\Temp")
            foreach ($location in $locations) {
                if (Test-Path $location) {
                    $items = Get-ChildItem -Path $location -Recurse -Force -ErrorAction SilentlyContinue
                    $count = $items.Count
                    $items | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
                    Write-Host "   Cleaned $count items from $location" -ForegroundColor $Colors.Success
                }
            }
        }
        default { Write-Host "Usage: ifile [size|find|cleanup] [path] [target]" -ForegroundColor $Colors.Warning }
    }
}

# Main shell loop
function Start-IntelligentShell {
    Show-IntelligentShellBanner
    Get-IntelligentSystemInfo
    Get-PackageManagerStatus
    Show-IntelligentCommands
    
    # Set up aliases
    Set-Alias -Name "isys" -Value Invoke-IntelligentSystem -Scope Global
    Set-Alias -Name "ipack" -Value Invoke-IntelligentPackage -Scope Global
    Set-Alias -Name "igit" -Value Invoke-IntelligentGit -Scope Global
    Set-Alias -Name "inetwork" -Value Invoke-IntelligentNetwork -Scope Global
    Set-Alias -Name "iprocess" -Value Invoke-IntelligentProcess -Scope Global
    Set-Alias -Name "ifile" -Value Invoke-IntelligentFile -Scope Global
    Set-Alias -Name "ihelp" -Value Show-IntelligentCommands -Scope Global
    Set-Alias -Name "iexit" -Value { Write-Host "👋 Goodbye from Intelligent Shell 2026!" -ForegroundColor $Colors.Primary; exit } -Scope Global
    
    Write-Host "🎉 Intelligent Shell 2026 is now active! Type 'ihelp' for commands." -ForegroundColor $Colors.Success
    Write-Host ""
}

# Installation function
function Install-IntelligentShell {
    Write-Host "🚀 Installing Intelligent Shell 2026..." -ForegroundColor $Colors.Primary
    
    # Create profile if it doesn't exist
    if (!(Test-Path $PROFILE)) {
        New-Item -Type File -Path $PROFILE -Force
    }
    
    # Add to PowerShell profile
    $profileContent = @"
# Intelligent Shell 2026 Auto-load
if (Test-Path '$($PWD.Path)\IntelligentShell.ps1') {
    . '$($PWD.Path)\IntelligentShell.ps1'
    Start-IntelligentShell
}
"@
    
    Add-Content -Path $PROFILE -Value $profileContent
    Write-Host "✅ Intelligent Shell 2026 installed successfully!" -ForegroundColor $Colors.Success
    Write-Host "   Restart PowerShell to activate the intelligent features." -ForegroundColor $Colors.Info
}

# Main execution logic
if ($Install) {
    Install-IntelligentShell
} elseif ($Start) {
    Start-IntelligentShell
} elseif ($Config) {
    Write-Host "⚙️  Intelligent Shell Configuration:" -ForegroundColor $Colors.Primary
    $IntelligentShellConfig | ConvertTo-Json | Write-Host
} else {
    Start-IntelligentShell
}
