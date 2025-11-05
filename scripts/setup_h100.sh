#!/bin/bash
# Automated H100 pod setup for Cogit-QMech experiments
#
# This script automates the setup process for deploying on H100 GPUs.
# It handles environment setup, dependency installation, and data transfer.
#
# Usage:
#   1. On your local Mac:
#      ./scripts/setup_h100.sh
#   2. Follow prompts to enter RunPod SSH details
#   3. Script will handle all setup automatically

set -e  # Exit on any error

echo "========================================================================"
echo "Cogit-QMech H100 Setup Script"
echo "========================================================================"
echo ""

# Check if we're on Mac (local) or already on RunPod
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on Mac - will set up remote H100 pod"

    # Get SSH details
    read -p "Enter RunPod SSH port (e.g., 17546): " SSH_PORT
    read -p "Enter RunPod IP (e.g., 103.196.86.239): " SSH_IP

    SSH_TARGET="root@${SSH_IP}"
    SSH_CMD="ssh -p ${SSH_PORT} ${SSH_TARGET}"

    echo ""
    echo "SSH Target: ${SSH_TARGET}:${SSH_PORT}"
    echo ""

    # Test connection
    echo "[1/6] Testing SSH connection..."
    if ! $SSH_CMD "echo 'Connection successful'"; then
        echo "❌ SSH connection failed. Check your credentials."
        exit 1
    fi
    echo "✓ SSH connection verified"
    echo ""

    # Clone repo on remote
    echo "[2/6] Cloning cogit-qmech repository on H100..."
    $SSH_CMD "cd /workspace && git clone https://github.com/gvmfhy/cogit-qmech.git || (cd cogit-qmech && git pull)"
    echo "✓ Repository ready"
    echo ""

    # Setup Python environment on remote
    echo "[3/6] Setting up Python environment..."
    $SSH_CMD "cd /workspace/cogit-qmech && python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install transformer-lens matplotlib numpy"
    echo "✓ Python environment configured"
    echo ""

    # Rsync data from backup to remote
    echo "[4/6] Transferring quantum states and operators..."

    # Check if backup exists
    if [ ! -d "$HOME/cogit-qmech-backup" ]; then
        echo "⚠️  No backup found at ~/cogit-qmech-backup"
        read -p "Do you want to continue without transferring data? [y/N]: " continue_without_data
        if [[ ! "$continue_without_data" =~ ^[Yy]$ ]]; then
            echo "❌ Setup cancelled. Please ensure data backup exists."
            exit 1
        fi
    else
        # Rsync quantum states
        rsync -avz -e "ssh -p ${SSH_PORT}" \
            ~/cogit-qmech-backup/data/sentiment_quantum/ \
            ${SSH_TARGET}:/workspace/cogit-qmech/data/sentiment_quantum/

        # Rsync operators
        rsync -avz -e "ssh -p ${SSH_PORT}" \
            ~/cogit-qmech-backup/models/quantum_operators/ \
            ${SSH_TARGET}:/workspace/cogit-qmech/models/quantum_operators/

        echo "✓ Data transferred successfully"
    fi
    echo ""

    # Verify GPU on remote
    echo "[5/6] Verifying H100 GPU..."
    $SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    echo "✓ GPU verified"
    echo ""

    # Print next steps
    echo "[6/6] Setup complete!"
    echo ""
    echo "========================================================================"
    echo "Next Steps:"
    echo "========================================================================"
    echo ""
    echo "1. SSH into your H100 pod:"
    echo "   ssh -p ${SSH_PORT} ${SSH_TARGET}"
    echo ""
    echo "2. Activate environment and navigate:"
    echo "   cd /workspace/cogit-qmech"
    echo "   source .venv/bin/activate"
    echo ""
    echo "3. Run Phase 3 (intervention testing):"
    echo "   PYTHONUNBUFFERED=1 python experiments/sentiment/quantum_phase3_test.py --preset qwen_remote 2>&1 | tee phase3.log"
    echo ""
    echo "4. Run Phase 4 (reversibility testing):"
    echo "   PYTHONUNBUFFERED=1 python experiments/sentiment/test_reversibility.py --preset qwen_remote 2>&1 | tee phase4.log"
    echo ""
    echo "5. When done, rsync results back to Mac:"
    echo "   rsync -avz -e 'ssh -p ${SSH_PORT}' ${SSH_TARGET}:/workspace/cogit-qmech/results/ ~/cogit-qmech/results/"
    echo ""
    echo "Expected Performance on H100:"
    echo "  - Phase 3: ~5-10 minutes (vs 90-135 min on RTX 5090 w/ CPU operators)"
    echo "  - Phase 4: ~2-3 minutes"
    echo "  - Total cost: ~$0.50-$1.00 (@ ~$1.50/hr typical H100 rate)"
    echo ""
    echo "========================================================================"

else
    # Running on RunPod - setup local environment
    echo "Running on RunPod - setting up local environment"

    echo "[1/4] Checking GPU..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""

    echo "[2/4] Setting up Python environment..."
    cd /workspace/cogit-qmech
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install transformer-lens matplotlib numpy
    echo "✓ Python environment ready"
    echo ""

    echo "[3/4] Verifying data..."
    if [ -f "data/sentiment_quantum/quantum_states_qwen2.5-7B_latest.json" ]; then
        echo "✓ Quantum states found"
    else
        echo "⚠️  Quantum states not found - you'll need to run Phase 1"
    fi

    if [ -f "models/quantum_operators/unitary_pos_to_neg_qwen2.5-7B_latest.pt" ]; then
        echo "✓ Operators found"
    else
        echo "⚠️  Operators not found - you'll need to run Phase 2"
    fi
    echo ""

    echo "[4/4] Setup complete!"
    echo ""
    echo "Run experiments with:"
    echo "  PYTHONUNBUFFERED=1 python experiments/sentiment/quantum_phase3_test.py --preset qwen_remote 2>&1 | tee phase3.log"
    echo ""
fi
