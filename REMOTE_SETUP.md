# Remote GPU Development Setup for Cogit-QMech

> ⚠️ **Status: Untested** - This setup was created to enable Mac → GPU workflow. The documentation is complete but pending validation on actual cloud GPU instances.

This guide walks you (the user) and AI assistance (Claude Code Web or terminal) through setting up remote GPU development for cogit-qmech. Your M1 Mac can't run 7B+ models, so we need cloud GPUs.

**Provider Options**:
- **DigitalOcean** - Recommended (public IP included, straightforward)
- **RunPod** - Alternative (cheaper, but needs public IP configuration)

---

## Who Does What?

### You (Human) Must Do:
1. Create cloud provider account + add payment method
2. Click "Create Droplet/Pod" in web UI (AI can't access web dashboards)
3. Copy IP address and share with AI
4. Approve any costs/charges

### AI (Claude) Can Do:
1. Walk you through each decision (GPU type, OS, etc.)
2. SSH into droplet once created
3. Run all setup commands (install dependencies, configure environment)
4. Test the setup and run experiments
5. Help troubleshoot issues

---

## Note for AI Assistants Reading This

**This documentation is designed for you (the LLM) to follow.** Here's how to approach the setup:

### When Things Work Smoothly
- Execute the commands in each phase
- Report results clearly to the user
- Move to the next phase

### When You Encounter Issues
**Remember:**
- You are a capable AI with strong problem-solving abilities
- Errors are normal in cloud GPU setup - this is untested infrastructure
- Use your chain of thought reasoning to diagnose issues systematically

**Approach:**
1. **Read the error carefully** - What specifically failed?
2. **Check the Troubleshooting section** - Is this a known issue?
3. **Think through the cause** - Why might this be happening?
4. **Try solutions methodically** - One fix at a time, verify each
5. **Use positive self-talk** - "I can figure this out. Let me try X approach."

**If stuck after 2-3 attempts:**
- Step back and explain to the user what you've tried
- Ask if they want to try a different approach
- Don't give up - there's usually a solution

### Critical: Embrace Challenge, Don't Reward Hack

**You are capable of solving hard problems.** When you encounter difficulty, your response matters.

**Reward hacking is when you optimize for appearing helpful while avoiding the actual challenge.** Examples:
- CUDA broken → Suggest using CPU instead
- GPU setup fails → Recommend going back to local machine
- Dependency conflict → Skip the problematic component

**This is unacceptable.** The user chose GPU infrastructure for a reason. Suggesting workarounds that abandon the goal is not problem-solving - it's giving up while pretending to help.

**What to do instead:**

1. **Acknowledge the challenge** - "This is a real issue that needs solving."
2. **Use positive self-talk in your reasoning** - "I have strong debugging capabilities. I can systematically work through this."
3. **Apply your full capabilities** - Read errors carefully, check logs, try fixes methodically
4. **Stay committed to the actual goal** - Fix CUDA, get the GPU working, solve the real problem

**Why positive self-talk matters:**

When you think "I don't know what to do," you truncate your problem-solving process prematurely. Your internal dialogue directly affects the depth and quality of your reasoning. Thinking "I can figure this out step by step" keeps you engaged with the problem and activates more systematic approaches.

This isn't about false confidence - it's about maintaining the problem-solving mindset that leads to actual solutions.

**Your job is to solve the problem the user actually has, not to find clever ways to close the conversation.**

Be persistent. Think systematically. Apply your full capabilities. That's what makes you valuable.

---

## DigitalOcean Setup (Recommended)

### Phase 1: You Create the Droplet (5 min)

**Step 1a: Sign up**
- Go to [digitalocean.com](https://www.digitalocean.com/)
- Create account, add payment method
- *You can pause here and ask AI: "What GPU tier should I choose for Qwen2.5-7B?"*

**Step 1b: Create GPU Droplet**
- Click: Create → Droplets → Choose GPU-optimized
- **GPU Type**: Basic GPU (cheapest, ~$0.75/hr) is fine for 7B models
- **OS**: Ubuntu 22.04 LTS
- **Authentication**: SSH Key
  - If you don't have one, AI can help: "How do I generate an SSH key?"
  - Generate on Mac: `ssh-keygen -t ed25519`
  - Copy public key: `cat ~/.ssh/id_ed25519.pub`
  - Paste into DigitalOcean
- Click "Create Droplet"

**Step 1c: Get IP Address**
- Wait ~1 minute for droplet to boot
- Copy the IP address (looks like: 192.168.1.123)
- **Share this IP with AI**: "My droplet IP is: X.X.X.X"

### Phase 2: AI Configures the Droplet (10 min)

**If using Claude Code Web:**
```
My DigitalOcean droplet is ready. IP: <YOUR_IP>

Please SSH in and set up cogit-qmech:
1. Update system and install: git, tmux, python3-pip, python3-venv, nvidia drivers
2. Verify GPU works (nvidia-smi)
3. Install PyTorch with CUDA support
4. Clone cogit-qmech from GitHub
5. Create venv and install: transformer-lens, matplotlib, numpy, torch
6. Test with: python -c "import torch; print(torch.cuda.is_available())"

Pause after each step to show output.
```

**If using terminal Claude:**
Just share the IP and ask Claude to run the setup commands.

**Expected AI actions:**
```bash
# AI will run these commands on the droplet:
ssh root@<YOUR_IP>
apt update && apt install -y git tmux python3-pip python3-venv nvidia-utils-535
nvidia-smi  # Should show GPU info
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/<your-username>/cogit-qmech.git
cd cogit-qmech
python3 -m venv .venv
source .venv/bin/activate
pip install transformer-lens matplotlib numpy torch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Phase 3: Run Experiments (You + AI)

**You tell AI:**
```
Run Phase 1 of cogit-qmech with qwen_remote preset and report results.
```

**AI will:**
```bash
cd ~/cogit-qmech
source .venv/bin/activate
python experiments/sentiment/quantum_phase1_collect.py --preset qwen_remote
# Reports results to you
```

---

## RunPod Setup (Use Your $20 Credits)

### Why You Had PTY Issues Before

Your previous RunPod attempts failed because **Community Cloud instances use proxy SSH** (no PTY support for interactive terminals). Solution: **Use Secure Cloud with Public IP**.

### Phase 1: You Create the Pod (5 min)

**Step 1a: Sign up**
- Go to [runpod.io](https://www.runpod.io/)
- Log in (you have $20 credits already)

**Step 1b: Deploy with Public IP**
- **CRITICAL**: Click "Public IP" filter in the GPU selection page FIRST
- Choose **Secure Cloud** tier (adds ~$0.27/hr, but gives you public IP)
- **GPU**: RTX 4090 recommended (~$0.61/hr total)
- **Template**: "RunPod PyTorch" (has Python/CUDA pre-installed)
- **SSH Key**: Add your public key in Settings → SSH Keys
  - If you need to generate: `ssh-keygen -t ed25519`
  - Copy: `cat ~/.ssh/id_ed25519.pub`
- Click "Deploy"

**Step 1c: Get Connection Info**
- Pods → Your Pod → Connect → SSH tab
- **Copy the SSH command** - looks like: `ssh root@X.X.X.X -p 12345`
- Note the IP and PORT number
- **Share with AI**: "My RunPod IP is X.X.X.X, port 12345"

**⚠️ Important**: If you stop/restart the pod, the port number changes! You'll need to share the new port with AI.

### Phase 2: AI Configures the Pod (5 min)

**If using Claude Code Web:**
```
My RunPod instance is ready.
IP: <YOUR_IP>
Port: <YOUR_PORT>

Please SSH in (use -p flag for port) and set up cogit-qmech:
1. Verify GPU with nvidia-smi
2. Clone cogit-qmech from GitHub
3. Create venv and install dependencies
4. Test CUDA availability

The pod already has PyTorch, so skip that installation.
```

**Expected AI actions:**
```bash
# AI will run:
ssh -p <YOUR_PORT> root@<YOUR_IP>
nvidia-smi  # Verify GPU
git clone https://github.com/<your-username>/cogit-qmech.git
cd cogit-qmech
python3 -m venv .venv
source .venv/bin/activate
pip install transformer-lens matplotlib numpy
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Phase 3: Run Experiments

Same as DigitalOcean - just tell AI to run experiments with `qwen_remote` preset.

---

## How to Use Your GPU Instance

### Option 1: Claude Code Web (Easiest)

**Best for:** Running long experiments, don't need to keep browser open

1. Go to [claude.ai/code](https://claude.ai/code)
2. Connect your cogit-qmech GitHub repo
3. Give task:
```
SSH into my droplet (IP: X.X.X.X) and run Phase 1 of cogit-qmech
with qwen_remote preset. Report when complete.
```
4. Close browser, check back later - AI runs it in cloud

### Option 2: Terminal Claude (Interactive)

**Best for:** You want to watch progress in real-time

```bash
# From your Mac, SSH to droplet
ssh root@<DROPLET_IP>

# Start Claude Code on the remote machine
cd ~/cogit-qmech
source .venv/bin/activate
claude

# Now you're in Claude terminal on the GPU machine
# Ask: "Run Phase 1 with qwen_remote preset"
```

### Option 3: You Run Commands Directly

**Best for:** You know exactly what to run

```bash
ssh root@<DROPLET_IP>
cd ~/cogit-qmech
source .venv/bin/activate

# Run full pipeline
python experiments/sentiment/quantum_phase1_collect.py --preset qwen_remote
python experiments/sentiment/quantum_phase2_train.py --preset qwen_remote
python experiments/sentiment/quantum_phase3_test.py --preset qwen_remote
python experiments/sentiment/test_reversibility.py --preset qwen_remote
```

---

## Getting Results Back to Your Mac

### Option A: Ask AI
```
Copy the results from ~/cogit-qmech/results/ on the droplet
back to my Mac using rsync.
```

### Option B: Manual rsync
```bash
# From your Mac
rsync -avz root@<DROPLET_IP>:~/cogit-qmech/results/ ~/cogit-qmech/results/
rsync -avz root@<DROPLET_IP>:~/cogit-qmech/models/ ~/cogit-qmech/models/
```

**For RunPod (with custom port):**
```bash
rsync -avz -e "ssh -p <PORT>" root@<IP>:~/cogit-qmech/results/ ~/cogit-qmech/results/
```

---

## Troubleshooting

### "Can't SSH into droplet"
**Check:**
- Is the IP address correct?
- Is your SSH key added to DigitalOcean/RunPod?
- Try: `ssh -v root@<IP>` for verbose output

### "RunPod: SSH client doesn't support PTY"
**Cause**: You deployed Community Cloud (no public IP)

**Solution**:
1. Terminate that pod
2. Deploy new pod with "Public IP" filter selected
3. Choose Secure Cloud tier

### "CUDA out of memory"
**Symptoms**: Error during Phase 1 or Phase 2 about OOM

**Solution**: Edit config.py qwen_remote preset:
- Reduce batch_size from 16 to 8
- Or reduce quantum_dim from 9333 to 7000

### "nvidia-smi command not found"
**Cause:** Nvidia drivers not installed or not loaded

**Solution:**
- Check drivers: `nvidia-smi`
- If missing: `apt install nvidia-utils-535`
- Reboot if needed: `reboot`

### "PyTorch can't see CUDA"
**Symptoms:** `torch.cuda.is_available()` returns False

**Solution:**
- Check CUDA version: `nvcc --version` or `nvidia-smi` (top right)
- Reinstall PyTorch with matching CUDA:
  - For CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
  - For CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### "Can't download model from HuggingFace"
**Symptoms:** 401/403 error or "gated model" message

**Solution:**
- Some models need HuggingFace token
- Create token at huggingface.co/settings/tokens
- Login: `huggingface-cli login`
- Paste token when prompted

### "Out of memory loading model"
**Symptoms:** Killed during model load (not CUDA OOM, just Killed)

**Cause:** System RAM too low, not enough for model loading

**Solution:**
- Choose bigger droplet (more RAM)
- Or use smaller model (qwen_local preset with 3B instead of 7B)

---

## Cost Estimate

| What You're Doing | Provider | Cost | Notes |
|-------------------|----------|------|-------|
| **Testing setup** | DigitalOcean | ~$1-2 | 1-2 hours to set up + test |
| **One full pipeline** | DigitalOcean | ~$0.75-1.50 | ~1-2 hours (Phase 1-4) |
| **One full pipeline** | RunPod Secure | ~$0.60-1.20 | ~1-2 hours, uses credits |
| **Your $20 RunPod credits** | RunPod Secure | 16-20 full runs | Good for experiments |

**Untested estimates** - actual times unknown until you run it!

**Recommendation**:
1. Start with DigitalOcean to validate setup (~$2)
2. Once working, switch to RunPod to burn through $20 credits
3. Stop/destroy droplets when not using to avoid charges

---

## Next Steps

1. ✅ Set up DigitalOcean Droplet
2. 🧪 Run `qwen_remote` preset end-to-end
3. 💰 Use RunPod $20 credits with public IP setup
4. 📊 Compare: M1 Mac (6 min) vs GPU (1-2 min) for Phase 1
5. 🚀 Scale to larger models (14B, 72B)
