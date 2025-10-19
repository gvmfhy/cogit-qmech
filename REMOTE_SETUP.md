# Remote GPU Setup Guide

Complete guide for setting up RunPod cloud GPU for Cogit-QMech experiments.

---

## Step 1: Create RunPod Account (5 min)

1. Go to https://runpod.io
2. Sign up (Google/GitHub OAuth recommended)
3. Add payment method (minimum $10 credit)
4. **Add your SSH public key**:
   - Go to Settings → SSH Keys
   - Click "Add SSH Key"
   - Paste this key:
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILZfd4VLv6xptdZPKK4ITmiy6bJgywTcn5nizG/c0qBe austinmorrissey@Austins-MacBook-Pro.local
   ```
   - Name it: "MacBook Pro M1"

---

## Step 2: Deploy GPU Pod (5 min)

1. Go to "Pods" → "Deploy"
2. **Filter GPUs**:
   - RTX A4000 (24GB VRAM) - ~$0.40/hr
   - RTX 4090 (24GB VRAM) - ~$0.50/hr
   - (Llama 7B needs ~14GB, these have headroom)

3. **Select Template**: "RunPod PyTorch 2.4" or "RunPod PyTorch"

4. **Configure Pod**:
   - Container Disk: 50GB
   - Volume: 50GB (persistent storage)
   - Expose TCP ports: 22 (SSH)
   - **Important**: Check "Expose HTTP Ports" for port 22

5. Click "Deploy On-Demand"

6. **Save connection details**:
   - SSH Command will look like: `ssh root@X.X.X.X -p XXXXX`
   - Note the IP and port number

---

## Step 3: Configure SSH on Your Mac (2 min)

Run this command to add RunPod to your SSH config:

```bash
cat >> ~/.ssh/config << 'EOF'

# RunPod GPU Instance
Host runpod-gpu
    HostName <REPLACE_WITH_POD_IP>
    Port <REPLACE_WITH_POD_PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF
```

**Replace**:
- `<REPLACE_WITH_POD_IP>` with the IP from RunPod (e.g., `45.67.89.123`)
- `<REPLACE_WITH_POD_PORT>` with the port (e.g., `12345`)

**Test connection**:
```bash
ssh runpod-gpu
```

You should see a RunPod welcome screen!

---

## Step 4: Setup Environment on Remote (5 min)

Once connected to the remote GPU, run:

```bash
# Clone the repo
git clone https://github.com/gvmfhy/cogit-qmech.git
cd cogit-qmech

# Run setup script
bash scripts/remote_setup.sh
```

The setup script will:
- ✓ Check GPU availability
- ✓ Install dependencies
- ✓ Test PyTorch CUDA
- ✓ Verify complex number support

---

## Step 5: Connect Cursor IDE (5 min)

1. Open Cursor on your Mac
2. Press `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
3. Select "runpod-gpu"
4. Wait for connection (30-60 seconds first time)
5. Open folder: `/root/cogit-qmech`

Now you can:
- ✅ Edit code on Mac (in Cursor)
- ✅ Run experiments on GPU (in Cursor's terminal)
- ✅ See results immediately
- ✅ Commit from remote or local

---

## Step 6: Run First Experiment

In Cursor's remote terminal:

```bash
cd /root/cogit-qmech
source .venv/bin/activate

# Test with GPT-2 first (sanity check)
python experiments/sentiment/quantum_phase1_collect.py --preset remote

# Then try Llama 7B (coming soon!)
```

---

## Daily Workflow

**Starting work**:
1. Start RunPod pod (if stopped)
2. Connect Cursor to `runpod-gpu`
3. Activate venv: `source .venv/bin/activate`

**During work**:
- Code in Cursor (saves automatically to remote)
- Run experiments in integrated terminal
- Monitor GPU usage: `nvidia-smi`

**Ending work**:
1. Commit your work: `git add . && git commit -m "..." && git push`
2. Stop the pod in RunPod dashboard (saves $$)

**Cost**: $0.40-0.50/hr × hours used

---

## Troubleshooting

**Connection refused**:
- Check pod is running in RunPod dashboard
- Verify IP/port in `~/.ssh/config` match current pod

**CUDA not available**:
- Run `nvidia-smi` to check GPU
- Restart pod if needed

**Dependencies missing**:
- Re-run `bash scripts/remote_setup.sh`

**Pod stopped automatically**:
- RunPod stops idle pods after 24h
- Save work to git before stopping!

---

## Cost Optimization

**Cheap GPUs** (occasional use):
- RTX A4000: $0.34/hr (secure cloud)
- RTX 4090: $0.44/hr (community cloud)

**Storage**:
- Container disk: Free (ephemeral)
- Volume disk: $0.10/GB/month (persistent)

**Tips**:
- Stop pod when not using (pause billing)
- Use "Spot" pricing for 50% discount (may be interrupted)
- Set auto-stop after 1 hour idle

---

## Next Steps

After setup works:
1. Test Llama 3.2 3B on GPU
2. Run full `remote` preset experiments
3. Investigate transfer attacks (7B → 70B operators)
