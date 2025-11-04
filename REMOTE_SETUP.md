# Remote GPU Development Setup for Cogit-QMech

> ⚠️ **Status: Untested** - This setup was created to enable Mac → GPU workflow. The documentation is complete but pending validation on actual cloud GPU instances.

This guide helps you overcome your M1 MacBook's computational bottleneck by setting up remote GPU development using **DigitalOcean** (primary, easiest) or **RunPod** (secondary, use $20 credits).

**Quick Decision Guide**:
- **Want it to just work?** → DigitalOcean (public IP included, zero PTY issues)
- **Have $20 RunPod credits?** → Follow RunPod section with public IP setup
- **Both?** → Set up DigitalOcean first, then RunPod for experiments

---

## Table of Contents

1. [DigitalOcean Setup (Recommended)](#digitalocean-setup)
2. [RunPod Setup (Use $20 Credits)](#runpod-setup)
3. [Using Claude Code & Cursor](#using-claude-code--cursor)
4. [Project Sync & Workflow](#project-sync--workflow)
5. [Troubleshooting](#troubleshooting)

---

## DigitalOcean Setup

### Why DigitalOcean?
- Public IP included (zero PTY/SSH issues)
- Works with Claude Code out-of-the-box
- Cursor Remote-SSH fully supported
- ~20% more expensive but worth it for reliability

### Quick Setup (15 min total)

**1. Create GPU Droplet** (5 min):
- Sign up at [DigitalOcean](https://www.digitalocean.com/)
- Create→ Droplets → GPU Droplet
- OS: Ubuntu 22.04 LTS
- Add your SSH key (or generate: `ssh-keygen -t ed25519`)
- Note the IP address after creation

**2. Initial Setup** (5 min):
```bash
# From Mac: SSH into droplet
ssh root@<DROPLET_IP>

# Update & install essentials
apt update && apt install -y git tmux python3-pip python3-venv nvidia-utils-535

# Verify GPU
nvidia-smi

# Install PyTorch with CUDA
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install Claude Code** (2 min):
```bash
# On the Droplet
curl -fsSL https://claude.ai/install.sh | sh
claude --version
```

**4. Sync & Setup Project** (3 min):
```bash
# From Mac: sync project
rsync -avz --exclude '.git' ~/cogit-qmech/ root@<DROPLET_IP>:~/cogit-qmech/

# On Droplet: setup venv
cd ~/cogit-qmech
python3 -m venv .venv
source .venv/bin/activate
pip install transformer-lens matplotlib numpy torch
```

Done! Now see [Using Claude Code & Cursor](#using-claude-code--cursor).

---

## RunPod Setup

### Important: Solving the PTY Issue

Your previous RunPod attempts failed because **community cloud instances lack PTY support**. Solution: **Use public IP instances**.

### Quick Setup with Public IP (15 min)

**1. Deploy with Public IP** (5 min):
- Login to [RunPod](https://www.runpod.io/)
- **CRITICAL**: Select **"Public IP"** filter BEFORE deploying
- Choose **Secure Cloud** tier (adds ~$0.27/hr for public IP)
- GPU: RTX 4090 (~$0.61/hr total)
- Template: "RunPod PyTorch"
- Add SSH key to account (Settings → SSH Keys)

**2. Get SSH Command** (2 min):
- Pods → Your Pod → Connect → SSH tab
- Copy command: `ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_ed25519`

**3. Add to SSH Config** (3 min):
```bash
# Edit ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'

# RunPod GPU with Public IP
Host runpod-gpu
    HostName <IP_FROM_RUNPOD>
    Port <PORT_FROM_RUNPOD>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
EOF
```

**4. Install & Sync** (5 min):
```bash
# SSH in
ssh runpod-gpu

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | sh

# From Mac: sync project
rsync -avz -e "ssh -p <PORT>" ~/cogit-qmech/ root@<IP>:~/cogit-qmech/
```

**⚠️ Port Changes**: If you stop/restart the pod, the port number changes. Update `~/.ssh/config` with the new port from RunPod dashboard.

---

## Using Claude Code & Cursor

### Option A: Direct SSH + Claude Code

```bash
# From Mac, SSH into remote
ssh root@<DROPLET_IP>  # or ssh runpod-gpu

# Start tmux for persistence
tmux new -s cogit

# Run Claude Code
cd ~/cogit-qmech
source .venv/bin/activate
claude
```

### Option B: Cursor IDE Remote-SSH (Recommended)

**Setup (one-time)**:
1. Install **Remote-SSH** extension in Cursor
2. `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
3. Select your host (DigitalOcean IP or `runpod-gpu`)
4. Open folder: `/root/cogit-qmech`

**Invoke Claude from Cursor**:
1. Open Cursor terminal (`` Ctrl+` ``)
2. Activate venv: `source .venv/bin/activate`
3. Run: `claude`

**If Claude Code hangs on RunPod** (SSH passphrase issue):
```bash
# Use passwordless key instead
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519_nopass
# Update RunPod config to use this key
```

---

## Project Sync & Workflow

### Automated Sync Script

Create `sync_to_remote.sh`:
```bash
#!/bin/bash
# Usage: ./sync_to_remote.sh <provider> <host> [port]

PROVIDER=$1
HOST=$2
PORT=${3:-22}

if [ "$PROVIDER" = "digitalocean" ]; then
    rsync -avz --exclude '.git' --exclude '__pycache__' \
      ~/cogit-qmech/ root@$HOST:~/cogit-qmech/
elif [ "$PROVIDER" = "runpod" ]; then
    rsync -avz -e "ssh -p $PORT" --exclude '.git' \
      ~/cogit-qmech/ root@$HOST:~/cogit-qmech/
fi
```

### Running Experiments with qwen_remote

```bash
# SSH into remote
cd ~/cogit-qmech
source .venv/bin/activate

# Run full pipeline with Qwen2.5-7B on GPU
python experiments/sentiment/quantum_phase1_collect.py --preset qwen_remote
python experiments/sentiment/quantum_phase2_train.py --preset qwen_remote
python experiments/sentiment/quantum_phase3_test.py --preset qwen_remote
python experiments/sentiment/test_reversibility.py --preset qwen_remote
```

### Pulling Results Back

```bash
# From Mac
rsync -avz root@<HOST>:~/cogit-qmech/results/ ~/cogit-qmech/results/
rsync -avz root@<HOST>:~/cogit-qmech/models/ ~/cogit-qmech/models/
```

---

## Troubleshooting

### RunPod: "SSH client doesn't support PTY"
**Cause**: Using community cloud without public IP

**Solution**: Deploy Secure Cloud with public IP filter selected

### Cursor: Can't connect to RunPod
**Check**:
1. Pod is running in dashboard
2. Port number in `~/.ssh/config` matches current pod
3. Template supports "SSH over exposed TCP"

### CUDA out of memory
**Solution**: Reduce batch size or quantum dimension in config.py

### Claude Code hangs in Cursor terminal
**Cause**: SSH passphrase issue

**Solution**: Use passwordless SSH key or run ssh-agent before Claude

---

## Cost Comparison

| Provider | GPU | Cost/hr | Full Pipeline | Notes |
|----------|-----|---------|---------------|-------|
| **M1 Mac** | - | $0 | ~10 min (3B only) | Bottlenecked |
| **DigitalOcean** | Basic GPU | ~$0.75 | ~8 min (7B) | Easy, reliable |
| **RunPod Secure** | RTX 4090 | ~$0.61 | ~6 min (7B) | +$0.27 for public IP |
| **RunPod Community** | RTX 4090 | ~$0.34 | N/A | No PTY support! |

**Recommendation**: Start with DigitalOcean, then use RunPod $20 credits for experiments.

---

## Next Steps

1. ✅ Set up DigitalOcean Droplet
2. 🧪 Run `qwen_remote` preset end-to-end
3. 💰 Use RunPod $20 credits with public IP setup
4. 📊 Compare: M1 Mac (6 min) vs GPU (1-2 min) for Phase 1
5. 🚀 Scale to larger models (14B, 72B)
