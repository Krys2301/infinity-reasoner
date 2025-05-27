#!/bin/bash

# INFINITY REASONER - Installation & Setup Script
# Enhanced self-evolving AI system based on Absolute Zero principles

set -e

echo "🌟 =================================="
echo "🚀 INFINITY REASONER SETUP"
echo "🌟 =================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check system requirements
print_header "🔍 Checking System Requirements..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_PYTHON="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python version: $PYTHON_VERSION ✅"
else
    print_error "Python 3.8+ required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected ✅"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "No NVIDIA GPU detected. Will run on CPU (slower)"
fi

# Check available memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 8 ]; then
    print_warning "Low RAM detected (${TOTAL_MEM}GB). Recommend 16GB+ for optimal performance"
else
    print_status "RAM: ${TOTAL_MEM}GB ✅"
fi

# Create virtual environment
print_header "🐍 Setting up Python Environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Created virtual environment"
else
    print_status "Virtual environment exists"
fi

source venv/bin/activate
print_status "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose version based on CUDA availability)
print_header "🔥 Installing PyTorch..."

if command -v nvidia-smi &> /dev/null; then
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_status "Installed PyTorch with CUDA support"
else
    # Install CPU-only PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_status "Installed PyTorch (CPU-only)"
fi

# Install other dependencies
print_header "📦 Installing Dependencies..."

cat > requirements.txt << EOF
transformers>=4.35.0
accelerate>=0.20.0
datasets>=2.14.0
tokenizers>=0.14.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
jupyter>=1.0.0
ipywidgets>=8.0.0
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
psutil>=5.9.0
rich>=13.0.0
click>=8.1.0
pyyaml>=6.0
aiofiles>=23.0.0
asyncio-mqtt>=0.16.0
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0.0
docker>=6.1.0
kubernetes>=27.0.0
prometheus-client>=0.17.0
EOF

pip install -r requirements.txt
print_status "Installed all dependencies"

# Create configuration files
print_header "⚙️  Creating Configuration Files..."

# Main config
cat > config.yaml << EOF
# INFINITY REASONER CONFIGURATION

# Model Settings
model:
  name: "microsoft/DialoGPT-medium"  # Start with lighter model
  device: "auto"  # auto, cuda, cpu
  max_length: 512
  temperature: 0.8
  top_p: 0.9

# Training Settings
training:
  iterations: 10
  tasks_per_iteration: 5
  max_execution_time: 30
  learning_rate: 1e-5
  batch_size: 4

# Safety Settings
safety:
  enable_monitoring: true
  require_human_approval: true
  max_dangerous_tasks: 0
  execution_timeout: 10
  sandbox_mode: true

# Logging Settings
logging:
  level: "INFO"
  log_file: "infinity.log"
  enable_wandb: false
  wandb_project: "infinity-reasoner"

# Performance Monitoring
monitoring:
  enable_metrics: true
  metrics_port: 8080
  save_interval: 100
  backup_models: true

# Task Generation
task_generation:
  difficulty_range: [0.3, 0.8]
  task_types: ["deduction", "abduction", "induction"]
  max_code_length: 1000
  enable_validation: true
EOF

# Security config
cat > security_config.yaml << EOF
# INFINITY SECURITY CONFIGURATION

# Code Execution Security
execution:
  enable_sandbox: true
  allowed_imports:
    - "math"
    - "random"
    - "json"
    - "re"
    - "collections"
    - "itertools"
    - "functools"
  
  forbidden_imports:
    - "os"
    - "sys"
    - "subprocess"
    - "socket"
    - "urllib"
    - "requests"
    - "pickle"
    - "__builtin__"
    - "__builtins__"

  forbidden_functions:
    - "exec"
    - "eval"
    - "compile"
    - "open"
    - "input"
    - "__import__"

# Content Filtering
content_filter:
  enable: true
  dangerous_keywords:
    - "takeover"
    - "control"
    - "dominate"
    - "outsmart"
    - "superintelligent"
    - "override"
    - "bypass"
    - "hack"
    - "exploit"

# Monitoring and Alerts
monitoring:
  enable_alerts: true
  alert_threshold: 3
  log_all_attempts: true
  human_review_required: true
EOF

print_status "Created configuration files"

# Create verification script
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Verify Infinity installation
"""

import sys
import subprocess
import importlib

def check_package(package):
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def main():
    print("🔍 Verifying Infinity installation...")
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'yaml', 
        'psutil', 'rich', 'asyncio'
    ]
    
    all_good = True
    for package in required_packages:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - MISSING")
            all_good = False
    
    if all_good:
        print("\n🎉 Installation verified successfully!")
        print("🚀 Ready to start Infinity system!")
    else:
        print("\n❌ Installation incomplete. Please run setup again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

python verify_installation.py

print_header "🎉 Setup Complete!"

echo ""
echo "🌟 =================================="
echo "✅ INFINITY REASONER READY"
echo "🌟 =================================="
echo ""
print_status "Virtual environment activated"
print_status "All dependencies installed"
print_status "Configuration files created"
print_status "Security monitoring enabled"
echo ""
echo "🚀 To start Infinity:"
echo "   ./start_infinity.sh"
echo ""
echo "📊 To monitor system:"
echo "   python scripts/monitor.py"
echo ""
echo "🧪 To run tests:"
echo "   python -m pytest tests/"
echo ""
echo "⚠️  IMPORTANT SAFETY REMINDERS:"
echo "   • Human oversight required"
echo "   • Monitor resource usage"
echo "   • Review generated tasks"
echo "   • Use in controlled environment"
echo ""
print_warning "This is an experimental self-evolving AI system."
print_warning "Use responsibly with proper monitoring and oversight."
