# Cogit-QMech Quick Start Guide

**Goal**: Run the complete quantum cognitive operator pipeline in 10 minutes.

---

## Prerequisites

- Python 3.8+
- 16GB RAM recommended (M1 MacBook Pro works great!)
- ~2GB disk space for data and models

---

## Step 1: Installation (2 minutes)

```bash
cd ~/cogit-qmech

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python config.py
```

**Expected output**: Configuration presets displayed with memory estimates.

---

## Step 2: Choose Your Preset

Three presets available:

| Preset | Quantum Dim | RAM | Time | Device | Use Case |
|--------|------------|-----|------|--------|----------|
| `tiny` | 500-d | ~0.5GB | 5 min | CPU | Quick testing |
| `local` | 2,000-d | ~0.6GB | 15 min | MPS (M1) | Local development |
| `remote` | 10,000-d | ~3.5GB | 10 min | CUDA | Cloud GPU |

**Recommendation**: Start with `tiny` for your first run, then use `local` for real experiments.

---

## Step 3: Phase 1 - Collect Quantum States (3 minutes)

```bash
# For quick test (5 minutes total)
python experiments/sentiment/quantum_phase1_collect.py --preset tiny

# For local M1 (15 minutes total)
python experiments/sentiment/quantum_phase1_collect.py --preset local
```

**What happens**:
1. Loads GPT-2
2. Generates/loads 50 positive + 50 negative prompts
3. Extracts activations from layer 6
4. Encodes as normalized complex quantum states
5. Saves to `data/sentiment_quantum/`

**Expected output**:
```
✓ Collected 10 positive quantum states
✓ Collected 10 negative quantum states
✓ States are normalized: ||ψ|| = 1
✓ Separation measured via quantum fidelity
```

---

## Step 4: Phase 2 - Train Unitary Operators (5 minutes)

```bash
# Match the preset from Phase 1
python experiments/sentiment/quantum_phase2_train.py --preset tiny
```

**What happens**:
1. Loads quantum states from Phase 1
2. Creates two unitary operators (U_pos→neg, U_neg→pos)
3. Trains with Born rule loss
4. Verifies unitarity: U†U = I
5. Tests reversibility
6. Saves operators to `models/quantum_operators/`

**Expected output**:
```
Epoch 50/50 | Loss: 0.234567 | Fidelity: 0.765433
✓ Training complete in 120.5s
✓ Both operators maintain unitarity: U†U = I
pos → neg → pos fidelity: 0.7821
```

---

## Step 5: Phase 3 - Test Interventions (2 minutes)

```bash
python experiments/sentiment/quantum_phase3_test.py --preset tiny
```

**What happens**:
1. Loads GPT-2, encoder, decoder, and trained operators
2. Tests on 5 neutral prompts
3. Applies interventions at different blend ratios
4. Generates text with/without quantum manipulation
5. Saves results to `results/quantum_intervention/`

**Example output**:
```
PROMPT: 'The meeting this afternoon will'

📝 Baseline (no intervention):
   → be productive and efficient

😞 With U_pos→neg (make more negative):
   Blend 0.05: be challenging and difficult
   Blend 0.10: be frustrating and problematic

🌟 With U_neg→pos (make more positive):
   Blend 0.05: be wonderful and successful
```

---

## Step 6: Phase 4 - Test Reversibility (1 minute)

**This is the KEY test that classical HDC cannot pass!**

```bash
python experiments/sentiment/test_reversibility.py --preset tiny
```

**What happens**:
1. Tests pos → neg → pos round-trip
2. Tests neg → pos → neg round-trip
3. Measures quantum fidelity
4. Creates visualization
5. Saves results and plot

**Expected output**:
```
[Test 1: Positive → Negative → Positive]
  Average fidelity: 0.8234 ± 0.0512
  ✅ GOOD reversibility

[Test 2: Negative → Positive → Negative]
  Average fidelity: 0.7891 ± 0.0623
  ✅ GOOD reversibility

✅ QUANTUM ADVANTAGE DEMONSTRATED!
   Good reversibility achieved.
   Classical HDC cannot achieve this!
```

---

## Complete Pipeline (One Command)

Run all phases sequentially:

```bash
# Quick test (10 minutes)
python experiments/sentiment/quantum_phase1_collect.py --preset tiny && \
python experiments/sentiment/quantum_phase2_train.py --preset tiny && \
python experiments/sentiment/quantum_phase3_test.py --preset tiny && \
python experiments/sentiment/test_reversibility.py --preset tiny
```

---

## Scaling to Full Size

### Local (M1 MacBook)

```bash
# Use 'local' preset (2000-d quantum states)
# Estimated time: ~30-40 minutes total
python experiments/sentiment/quantum_phase1_collect.py --preset local
python experiments/sentiment/quantum_phase2_train.py --preset local
python experiments/sentiment/quantum_phase3_test.py --preset local
python experiments/sentiment/test_reversibility.py --preset local
```

### Cloud GPU (Full Scale)

For the full 10,000-d quantum version:

1. **Set up remote machine** (RunPod, Modal, etc.)
2. **Copy remote scripts** from classical repo:
   ```bash
   cp -r ../project-cogit-framework/scripts/remote scripts/
   ```
3. **Sync project**:
   ```bash
   scripts/remote/rsync_project.sh --alias runpod --remote-path /workspace/cogit-qmech
   ```
4. **Run remotely**:
   ```bash
   scripts/remote/remote_exec.sh --alias runpod --remote-path /workspace/cogit-qmech -- \
     "python experiments/sentiment/quantum_phase2_train.py --preset remote"
   ```

---

## Troubleshooting

### "No quantum states found"
- Run Phase 1 first before Phase 2/3/4
- Check `data/sentiment_quantum/` directory exists

### "Operator not found"
- Run Phase 2 before Phase 3/4
- Check `models/quantum_operators/` directory

### "Out of memory"
- Use smaller preset: `remote` → `local` → `tiny`
- Close other applications
- Use `preset=tiny` with `quantum_dim=500`

### "MPS backend not available"
- Your Mac isn't M1/M2/M3
- Change `device='mps'` to `device='cpu'` in config.py

### "Import errors"
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

---

## Understanding the Results

### Phase 1: Separation Analysis
```
Centroid fidelity: 0.2456 (lower is better, range [0,1])
Cross-class fidelity: 0.3123 (should be < within-class)
```
- **Good separation**: Centroid fidelity < 0.5
- Quantum states are distinguishable

### Phase 2: Training Metrics
```
Final fidelity: 0.7654
Unitary: True (deviation: 0.000002)
```
- **Good training**: Fidelity > 0.7
- **Unitarity maintained**: Deviation < 0.0001

### Phase 3: Intervention Quality
- **Coherence**: Text should remain grammatical
- **Sentiment shift**: Observe emotional tone changes
- **Blend ratio**: Lower is gentler (0.02-0.05 ideal)

### Phase 4: Reversibility
```
Average reversibility: 0.8123
✅ QUANTUM ADVANTAGE CONFIRMED!
```
- **Excellent**: > 0.9
- **Good**: > 0.7
- **Moderate**: > 0.5
- **Poor**: < 0.5

**Classical HDC typically gets < 0.3 here!**

---

## Next Steps

After running the quick start:

1. **Compare to classical HDC**
   - Run classical version on same prompts
   - Compare reversibility (quantum should win)
   - Compare intervention strength needed

2. **Experiment with hyperparameters**
   - Try different blend ratios
   - Adjust learning rates
   - Train for more epochs

3. **Analyze quantum properties**
   - Visualize complex amplitudes
   - Study entanglement (future work)
   - Investigate phase information

4. **Scale up**
   - Use `local` preset on M1
   - Use `remote` preset on cloud GPU
   - Test on different models (GPT-2 Medium/Large)

---

## Key Files to Inspect

- `results/quantum_intervention/quantum_results_latest.json` - Intervention results
- `results/quantum_intervention/reversibility_plot_*.png` - Reversibility visualization
- `models/quantum_operators/unitary_*_latest.pt` - Trained operators
- `data/sentiment_quantum/quantum_states_latest.json` - Encoded states

---

## Getting Help

- Check `STATUS.md` for implementation details
- Review `README.md` for theoretical background
- See classical version: `../project-cogit-framework/`
- File issues: [GitHub Issues](https://github.com/[your-repo]/issues)

---

## Success Criteria

You've successfully run Cogit-QMech if:

✅ All four phases complete without errors
✅ Reversibility > 0.7 (demonstrating quantum advantage)
✅ Interventions produce coherent text
✅ Operators maintain unitarity (U†U = I)

Congratulations! You've built and tested quantum cognitive operators. 🎉
