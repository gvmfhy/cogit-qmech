#!/bin/bash
# Remote GPU Setup Script for Cogit-QMech
# Run this on the RunPod instance after cloning the repo

set -e  # Exit on error

echo "========================================================================"
echo "🚀 Cogit-QMech Remote GPU Setup"
echo "========================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "config.py" ]; then
    echo "❌ Error: Run this from the cogit-qmech directory!"
    echo "   cd cogit-qmech && bash scripts/remote_setup.sh"
    exit 1
fi

# Check GPU availability
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi
echo ""

# Check Python version
echo "[2/6] Checking Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python 3.9+ detected"
else
    echo "❌ Error: Python 3.9+ required"
    exit 1
fi
echo ""

# Create virtual environment
echo "[3/6] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate and upgrade pip
echo "[4/6] Activating environment and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Install dependencies
echo "[5/6] Installing dependencies..."
echo "   This may take 2-3 minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install transformer-lens numpy matplotlib --quiet
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "[6/6] Verifying installation..."

# Check PyTorch CUDA
python3 << 'EOF'
import torch
import sys

print("\nPyTorch Configuration:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test complex number support
    device = torch.device('cuda')
    test_complex = torch.complex(torch.randn(10), torch.randn(10)).to(device)
    print(f"  Complex tensors on GPU: ✓")
else:
    print("  ⚠️  CUDA not available - will use CPU")

print("\n✓ PyTorch verification complete")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "🎉 Setup Complete!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Test with tiny preset:"
    echo "     python experiments/sentiment/quantum_phase1_collect.py --preset tiny"
    echo "  3. Run remote experiments:"
    echo "     python experiments/sentiment/quantum_phase1_collect.py --preset remote"
    echo ""
    echo "Monitor GPU usage:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
else
    echo ""
    echo "❌ Verification failed. Check errors above."
    exit 1
fi
