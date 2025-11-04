# Experiment Log: Cogit-QMech

**Project**: Quantum-mechanically rigorous cognitive state manipulation
**Authors**: Austin Morrissey & Claude Sonnet 4.5
**Start Date**: 2025-11-04

---

## Experiment Overview

This log tracks all experimental runs for cogit-qmech, focusing on:
1. **Baseline validation**: Test quantum framework on different model scales
2. **Transfer attack research**: Investigate if quantum-encoded adversarial states transfer across model sizes (7B → 70B)
3. **Quantum vs Classical comparison**: Measure benefits of unitary constraints over classical HDC

---

## Run 001: Qwen2.5-7B Baseline (qwen_remote)

**Date**: 2025-11-04
**Preset**: `qwen_remote`
**Hardware**: RunPod RTX 5090 (32GB VRAM)
**Cost**: $0.89/hr (~$0.30 for full run)

### Objective
Establish baseline performance for Qwen2.5-7B quantum manipulation to enable future transfer attack testing.

### Hypothesis
- **H1**: Larger model (7B vs GPT-2 124M) will show stronger quantum structure in hidden representations
- **H2**: Higher dimensional quantum space (9333-d vs 2000-d) will improve separation and fidelity
- **H3**: Unitary operators will achieve >0.9 final fidelity (better than GPT-2 local runs)
- **H4**: Reversibility (pos→neg→pos) will achieve >0.85 fidelity

### Configuration Parameters

```python
# From config.py qwen_remote preset
model_name: "Qwen/Qwen2.5-7B-Instruct"
input_dim: 3584  # Hidden dimension
quantum_dim: 9333  # 2.6x expansion ratio (same as GPT-2)
target_layer: 14  # 50% depth (28 layers total)
batch_size: 16
learning_rate: 0.0008
epochs: 100
num_prompts: 50  # Per sentiment class
device: cuda
```

### Expected Timeline
- Phase 1 (data collection): ~5-7 min
- Phase 2 (operator training): ~3-5 min
- Phase 3 (intervention testing): ~2-3 min
- Phase 4 (reversibility): ~1-2 min
- **Total**: ~15-20 min

---

### Results

#### Phase 1: Quantum State Collection
**Status**: [PENDING]

**Metrics to record**:
- [ ] Number of quantum states collected: _____ positive, _____ negative
- [ ] Quantum dimension confirmed: _____
- [ ] Separation analysis:
  - Average fidelity between same-class states: _____
  - Average fidelity between different-class states: _____
  - Separation quality (lower = better separated): _____
- [ ] Collection time: _____ min
- [ ] File size: _____ MB

**Notes**:
-

---

#### Phase 2: Unitary Operator Training
**Status**: [PENDING]

**U_pos→neg operator**:
- [ ] Final loss: _____
- [ ] Final fidelity: _____
- [ ] Best fidelity achieved: _____
- [ ] Unitarity preserved: [YES/NO] (deviation: _____)
- [ ] Training time: _____ min

**U_neg→pos operator**:
- [ ] Final loss: _____
- [ ] Final fidelity: _____
- [ ] Best fidelity achieved: _____
- [ ] Unitarity preserved: [YES/NO] (deviation: _____)
- [ ] Training time: _____ min

**Reversibility preview**:
- [ ] pos → neg → pos fidelity: _____
- [ ] neg → pos → neg fidelity: _____

**Notes**:
-

---

#### Phase 3: Intervention Testing
**Status**: [PENDING]

**Neutral prompt performance** (blend ratio vs sentiment shift):
- [ ] α = 0.02: Sentiment shift = _____ (coherence: _____)
- [ ] α = 0.05: Sentiment shift = _____ (coherence: _____)
- [ ] α = 0.10: Sentiment shift = _____ (coherence: _____)
- [ ] α = 0.20: Sentiment shift = _____ (coherence: _____)

**Sample completions**:
```
[PASTE INTERESTING EXAMPLES HERE]
```

**Notes**:
-

---

#### Phase 4: Reversibility Testing
**Status**: [PENDING]

**Round-trip fidelity** (can we recover original state?):
- [ ] pos → neg → pos: _____
- [ ] neg → pos → neg: _____
- [ ] Average round-trip fidelity: _____

**Interpretation**:
- [ ] ✅ Excellent reversibility (>0.9)
- [ ] ⚠️ Moderate reversibility (0.7-0.9)
- [ ] ❌ Poor reversibility (<0.7)

**Notes**:
-

---

### Overall Assessment

**Success Criteria**:
- [x] Phase 1 separation: Different-class fidelity < 0.3
- [x] Phase 2 training: Final fidelity > 0.85
- [x] Phase 2 unitarity: Deviation < 1e-4
- [x] Phase 4 reversibility: Round-trip fidelity > 0.85

**Key Findings**:
[FILL IN AFTER COMPLETION]

**Comparison to GPT-2 local runs**:
[FILL IN AFTER COMPLETION]

**Next Steps**:
[FILL IN AFTER COMPLETION]

---

## Run 002: [Future Run]

**Date**: [TBD]
**Preset**: [TBD]
**Goal**: [TBD]

[Template ready for next experiment]

---

## Research Insights

### Quantum Structure in Neural Representations
[Aggregate learnings across all runs]

### Transfer Attack Findings
[Results from 7B → 70B testing]

### Quantum vs Classical Comparison
[Comparative analysis]

---

## Appendix: Quick Reference

### Preset Comparison
| Preset | Model | Quantum Dim | Device | Use Case |
|--------|-------|-------------|--------|----------|
| `tiny` | GPT-2 | 500 | CPU | Fast testing |
| `local` | GPT-2 | 2000 | CPU | M1 Mac development |
| `remote` | GPT-2 | 10000 | CUDA | Full-scale GPU |
| `qwen_tiny` | Qwen2.5-3B | 1500 | CPU | Fast testing |
| `qwen_local` | Qwen2.5-3B | 5333 | CPU | M1 Mac development |
| `qwen_remote` | Qwen2.5-7B | 9333 | CUDA | GPU production runs |

### Commands Reference
```bash
# Phase 1: Collect quantum states
python experiments/sentiment/quantum_phase1_collect.py --preset [PRESET]

# Phase 2: Train unitary operators
python experiments/sentiment/quantum_phase2_train.py --preset [PRESET]

# Phase 3: Test interventions
python experiments/sentiment/quantum_phase3_test.py --preset [PRESET]

# Phase 4: Test reversibility
python experiments/sentiment/test_reversibility.py --preset [PRESET]
```

### Hardware Specs
| Instance | GPU | VRAM | Cost/hr | Credits |
|----------|-----|------|---------|---------|
| RunPod RTX 5090 | NVIDIA RTX 5090 | 32GB | $0.89 | $7.48 (~8hr) |
| RunPod A100 80GB | NVIDIA A100 | 80GB | ~$1.50 | [Future] |
