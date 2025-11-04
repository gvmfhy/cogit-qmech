# Troubleshooting Guide: Preventing Progress Loss

**Date**: 2025-11-04
**Lesson**: Never restart experiments without checking for existing progress

---

## Critical Rule: Check Before Restarting

**BEFORE restarting ANY phase:**

```bash
# 1. Check what files exist
ls -lth data/sentiment_quantum/quantum_states_*.json | head -3
ls -lth models/quantum_operators/unitary_*.pt | head -3
ls -lth results/quantum_intervention/*.json | head -3

# 2. Check timestamps match
# - Do Phase 2 models match latest Phase 1 quantum states?
# - Compare timestamps in filenames

# 3. Check model quality
python3 -c "
import torch
model = torch.load('models/quantum_operators/unitary_pos_to_neg_qwen2.5-7B_latest.pt', weights_only=False)
print('Timestamp:', model['timestamp'])
print('Final fidelity:', model['training_history']['fidelity_history'][-1])
print('Config:', model['config'])
"
```

**Decision tree:**
- ✅ **Models exist with good fidelity (>0.85) AND match current data** → USE THEM, skip to next phase
- ⚠️ **Models exist but timestamps don't match** → ASK USER if they want to re-train or use existing
- ❌ **Models don't exist or fidelity poor (<0.70)** → Safe to run phase

---

## Critical Rule: Document What Gets Thrown Away

When restarting a phase that invalidates downstream work:

**Required communication:**
```
⚠️ WARNING: Re-running Phase 1 will invalidate existing Phase 2 models

Current state:
- Phase 2 models trained at 23:12
- Final fidelity: 0.989 (98.9%)
- Will be invalidated if we re-run Phase 1

Options:
A) Use existing data (Phase 1 from 23:08 + Phase 2 from 23:12) → Proceed to Phase 3 immediately
B) Re-run Phase 1 with diverse prompts → Will need to re-run Phase 2 (~7 min)

Which do you prefer?
```

**Never assume** - always ask the user to make the call.

---

## Checkpointing Best Practices

### Why Checkpointing Matters

Without checkpoints:
- If training crashes at epoch 95/100 → restart from 0
- If you kill process thinking it's stuck → restart from 0
- Wasted GPU time = wasted money

### When to Add Checkpoints

**Short runs (<10 min)**: Optional
**Medium runs (10-30 min)**: Recommended every 10-20 epochs
**Long runs (>30 min)**: Required every 5-10 epochs

### How to Add Checkpointing to Phase 2

Modify `quantum_phase2_train.py`:

```python
# Inside training loop, after each epoch:
if (epoch + 1) % 10 == 0:  # Save every 10 epochs
    checkpoint_path = models_dir / f"checkpoint_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': operator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'fidelity_history': fidelity_history,
    }, checkpoint_path)
```

Resume from checkpoint:

```python
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    operator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_history = checkpoint['loss_history']
    fidelity_history = checkpoint['fidelity_history']
```

---

## Output Buffering Issues

### The Problem

Python buffers print() output when running via SSH. Training runs successfully but you can't see progress.

### Symptoms
- Process shows 100% GPU/CPU usage
- No terminal output for minutes
- Appears frozen but is actually working

### Solutions

**Option 1: Environment variable (preferred)**
```bash
PYTHONUNBUFFERED=1 python experiments/sentiment/quantum_phase2_train.py --preset qwen_remote
```

**Option 2: Python flag**
```bash
python -u experiments/sentiment/quantum_phase2_train.py --preset qwen_remote
```

**Option 3: Log to file**
```bash
python experiments/sentiment/quantum_phase2_train.py --preset qwen_remote > phase2.log 2>&1 &
tail -f phase2.log  # Monitor in real-time
```

**Option 4: Update code (add to all print statements)**
```python
print("Progress update", flush=True)
```

### Quick Check if Process is Actually Running

```bash
# Check GPU usage
nvidia-smi

# Check process CPU
ps aux | grep python | grep phase

# If both show high usage → training is working, output is just buffered
```

---

## Data Mismatch Prevention

### Symptom
Phase N uses data from Phase N-1, but timestamps don't match.

### Prevention

**Add to each phase script:**

```python
def verify_data_freshness(phase_name, required_files, max_age_minutes=30):
    """Verify input data is recent enough to be trustworthy"""
    from datetime import datetime
    import time

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"{phase_name} requires {file_path}")

        # Check file age
        file_time = os.path.getmtime(file_path)
        age_minutes = (time.time() - file_time) / 60

        if age_minutes > max_age_minutes:
            print(f"⚠️  WARNING: {file_path.name} is {age_minutes:.1f} minutes old")
            print(f"   This may not match current experimental state")
            response = input(f"   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                raise ValueError(f"User rejected stale data: {file_path.name}")

# Usage in Phase 2:
verify_data_freshness(
    "Phase 2",
    [data_dir / f"quantum_states_{model_id}_latest.json"],
    max_age_minutes=30
)
```

---

## Experiment State Management

### Create state file to track progress

**experiments/state.json:**
```json
{
  "current_run": "Run_001_Qwen7B",
  "phase_1": {
    "completed": true,
    "timestamp": "20251104_231942",
    "num_states": 50,
    "separation_fidelity": 0.9999
  },
  "phase_2": {
    "completed": true,
    "timestamp": "20251104_231202",
    "final_fidelity": 0.989,
    "matches_phase_1": false  # ← THIS WOULD HAVE CAUGHT IT
  },
  "phase_3": {
    "completed": false
  }
}
```

Each phase updates this file. Phase 2 checks: "Does my input match Phase 1 timestamp?"

---

## Summary: Three Rules to Prevent Progress Loss

1. **CHECK FIRST, ACT SECOND**
   - Always check what exists before restarting
   - Compare timestamps between phases
   - Verify data compatibility

2. **ASK, DON'T ASSUME**
   - If existing work will be invalidated, ask the user
   - Present options clearly (use existing vs re-run)
   - Never throw away >85% fidelity models without asking

3. **LOG EVERYTHING**
   - Save progress to files (checkpoints, logs, state.json)
   - Make timestamps visible in filenames
   - Document what was thrown away and why

---

**The mistake on 2025-11-04:** Restarted Phase 2, throwing away 98.9% fidelity models without asking. This guide ensures it never happens again.

---

## Performance Monitoring & Timing

### Why Track Timing

**Critical for:**
1. Cost estimation (GPU costs $/hour)
2. Diagnosing performance issues
3. Comparing model scales (7B vs 70B)
4. Identifying bottlenecks

### Required Timing Data

**Track for each phase:**
```
- Start time (timestamp)
- End time (timestamp)
- Duration (minutes)
- Hardware (CPU/GPU, model name)
- Key metrics (states collected, fidelity achieved, etc.)
```

### How to Track

**Add to each phase script:**

```python
import time
from datetime import datetime

start_time = time.time()
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[Phase N started at {start_timestamp}]")

# ... do work ...

end_time = time.time()
duration_min = (end_time - start_time) / 60

print(f"\n[Phase N completed in {duration_min:.1f} minutes]")

# Save to results
timing_data = {
    'start': start_timestamp,
    'duration_minutes': duration_min,
    'hardware': get_hardware_info()
}
```

### Performance Baseline: Qwen2.5-7B on RunPod RTX 5090

**Measured on 2025-11-04:**

| Phase | Task | Duration | Hardware | Notes |
|-------|------|----------|----------|-------|
| 1 | Data collection (50 prompts) | ~1 min | CPU | Would be faster on GPU |
| 2 | Train 2 operators (100 epochs each) | ~11 min | RTX 5090 GPU | 174M params each |
| 3 | Test interventions | ~15 min | CPU | Multiple inference passes |
| 4 | Reversibility test | ~2 min | CPU | |
| **Total** | Full pipeline | **~29 min** | Mixed | **$0.43 @ $0.89/hr** |

**Comparison to GPT-2 (124M) on M1 Mac:**
- Phase 1-4: ~30 min (estimated)
- Qwen2.5-7B is 60x larger but takes similar time due to better GPU

### Bottleneck Analysis

**Slow**: Phase 3 intervention testing
- **Why**: CPU inference on 7B model
- **Fix**: Move inference to GPU (requires TransformerLens GPU support)

**Fast**: Phase 2 training
- **Why**: GPU-accelerated, well-parallelized

**Acceptable**: Phase 1 data collection
- **Why**: One-time cost, CPU is fine

### Cost Tracking

**Always calculate:**
```
Cost = (Duration in hours) × (Hardware $/hr)

Example:
29 min = 0.48 hr
0.48 hr × $0.89/hr = $0.43 per full run
```

**Document in EXPERIMENT_LOG.md after each run.**
