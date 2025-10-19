# Quick Reference Card

---

## Your SSH Public Key (for RunPod)
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILZfd4VLv6xptdZPKK4ITmiy6bJgywTcn5nizG/c0qBe austinmorrissey@Austins-MacBook-Pro.local
```

---

## SSH Config Template
Add to `~/.ssh/config`:
```bash
Host runpod-gpu
    HostName <POD_IP_HERE>
    Port <POD_PORT_HERE>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

---

## Quick Commands

### Connect to Remote
```bash
ssh runpod-gpu
```

### On Remote: Setup
```bash
git clone https://github.com/gvmfhy/cogit-qmech.git
cd cogit-qmech
bash scripts/remote_setup.sh
```

### On Remote: Run Experiments
```bash
source .venv/bin/activate
python experiments/sentiment/quantum_phase1_collect.py --preset remote
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

---

## Cursor Remote SSH

1. `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
2. Select `runpod-gpu`
3. Open folder: `/root/cogit-qmech`

---

## Costs

- **RTX A4000**: $0.34-0.40/hr (24GB VRAM)
- **RTX 4090**: $0.44-0.50/hr (24GB VRAM)
- **Storage**: $0.10/GB/month

**Stop pod when not using to save money!**

---

## Links

- **RunPod**: https://runpod.io
- **GitHub Repo**: https://github.com/gvmfhy/cogit-qmech
- **Full Setup Guide**: See `REMOTE_SETUP.md`
