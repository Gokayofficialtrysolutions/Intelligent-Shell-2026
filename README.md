# Intelligent Shell 2026 🚀

**AI-Powered Terminal Enhancement for Windows PowerShell**  
*by Gokaytrysolutions*

[![PowerShell](https://img.shields.io/badge/PowerShell-5.0%2B-blue.svg)](https://github.com/PowerShell/PowerShell)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](releases)

## 🌟 Overview

Intelligent Shell 2026 is a comprehensive PowerShell enhancement that transforms your terminal experience with AI-powered features, intelligent command suggestions, and integrated system management tools.

## ✨ Features

- **🤖 AI-Powered Commands**: Intelligent command completion and suggestions
- **📊 System Monitoring**: Real-time CPU, memory, and system performance monitoring
- **📦 Package Management**: Unified interface for Chocolatey, Winget, and Scoop
- **🔧 Git Integration**: Enhanced git operations with visual status indicators
- **🌐 Network Tools**: Built-in network diagnostics and connectivity testing
- **⚡ Process Management**: Advanced process monitoring and control
- **📁 Smart File Operations**: Intelligent file search, analysis, and cleanup
- **🎨 Beautiful Interface**: Colorful, emoji-enhanced terminal experience

## 🚀 Quick Start

### Prerequisites
- Windows 10/11
- PowerShell 5.0 or higher
- Git (recommended)

### Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/Gokayofficialtrysolutions/Intelligent-Shell-2026.git
   cd Intelligent-Shell-2026
   ```

2. **Run the installer:**
   ```powershell
   .\Install.ps1
   ```

3. **Restart PowerShell** - The Intelligent Shell will start automatically!

### Manual Installation

If you prefer manual installation:

```powershell
# Load the shell manually
. .\IntelligentShell.ps1 -Start

# Or install to your PowerShell profile
. .\IntelligentShell.ps1 -Install
```

## 🛠️ Available Commands

| Command | Description | Example |
|---------|-------------|----------|
| `isys` | System information and monitoring | `isys info`, `isys monitor` |
| `ipack` | Package management | `ipack install nodejs`, `ipack list` |
| `igit` | Enhanced git operations | `igit status`, `igit quick "commit message"` |
| `inetwork` | Network diagnostics | `inetwork test`, `inetwork info` |
| `iprocess` | Process management | `iprocess top`, `iprocess memory` |
| `ifile` | File operations | `ifile size`, `ifile find "*.txt"` |
| `ihelp` | Show help menu | `ihelp` |
| `iexit` | Exit Intelligent Shell | `iexit` |

## 📊 Command Examples

### System Monitoring
```powershell
# Get system information
isys info

# Start real-time system monitor
isys monitor

# Clean temporary files
isys cleanup
```

### Package Management
```powershell
# Install a package (tries all available package managers)
ipack install git

# Search for packages
ipack search nodejs

# List installed packages
ipack list
```

### Git Integration
```powershell
# Enhanced git status with emojis
igit status

# Quick commit and push
igit quick "Fixed bug in login"

# Repository information
igit info
```

### Network Tools
```powershell
# Test connectivity
inetwork test google.com

# Network speed test
inetwork speed

# Show network interfaces
inetwork info
```

## ⚙️ Configuration

The shell uses `config.json` for customization:

```json
{
  "settings": {
    "autoStart": true,
    "showBanner": true,
    "colorScheme": "default"
  },
  "colors": {
    "primary": "Cyan",
    "secondary": "Green",
    "warning": "Yellow"
  }
}
```

## 📁 Project Structure

```
Intelligent-Shell-2026/
├── IntelligentShell.ps1    # Main shell script
├── Install.ps1             # Installation script
├── config.json             # Configuration file
├── README.md               # This file
└── .git/                   # Git repository
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all the PowerShell community for inspiration
- Special thanks to package manager developers (Chocolatey, Winget, Scoop)
- Emoji icons from various open source projects

## 📞 Support

For support, please:
- 📧 Open an issue on GitHub
- 💬 Join our discussions
- 🌟 Star the repository if you find it useful!

---

**Made with ❤️ by Gokaytrysolutions**  
*Enhancing your terminal experience one command at a time*
