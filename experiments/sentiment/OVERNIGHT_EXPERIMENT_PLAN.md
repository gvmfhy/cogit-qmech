# Overnight Experiment Plan: Improved Prompts Validation
**Date**: November 5, 2025
**Goal**: Validate hypothesis that neutral-stem prompts fix null steering
**For**: Research mentor meeting tomorrow

## Hypothesis Being Tested
**Root Cause**: Train/test distribution mismatch caused null steering
**Original Problem**: Operators trained on sentiment-biased prompts ("absolutely incredible and") don't generalize to neutral test prompts ("The weather today")
**Fix**: Use neutral-stem prompts that allow authentic model continuations

## Experiments Running

### Experiment 1: Qwen2.5-3B with Improved Prompts ✅ LAUNCHED
**Command**:
```bash
python experiments/sentiment/quantum_phase1_collect.py \
  --preset qwen_local \
  --prompts prompts/improved_prompts_100.json
```

**What it tests**: Does fixing prompts restore quantum state separation?
**Expected outcome**: Phase 1 separation gap ≥5%
**Log**: `qwen3b_improved_phase1.log`

**Then run full evaluation**:
```bash
python experiments/sentiment/evaluate_quantum_intervention.py \
  --preset qwen_local \
  --num-prompts 100 \
  --prompts prompts/improved_prompts_100.json \
  --neutral-prompts prompts/neutral_prompts_40.json \
  --blend-ratio 0.5
```

### Experiment 2: Blend Ratio Sweep (PENDING - after Exp 1)
**What it tests**: Is α=0.05 (5%) too weak? Literature uses 50-500%
**Ratios to test**: 0.05, 0.1, 0.5, 1.0
**Decision criteria**:
- ✅ PASS: ≥±10pp lift with 95% CI excluding 0
- ❌ FAIL: CI includes 0 (no effect)

### Experiment 3: Pythia-410M Comparison (PENDING)
**What it tests**: Does model size matter?
**Comparison**: Pythia-410M vs Qwen2.5-3B (both with improved prompts)
**Hypothesis**: Smaller models may have weaker quantum representations

### Experiment 4: Decode Method Comparison (OPTIONAL)
**What it tests**: Is real_component decoding suboptimal?
**Methods**:
- `real_component` (baseline): Uses only Re(ψ)
- `real_imag_avg`: Uses (Re(ψ) + Im(ψ))/2

## Success Criteria

### ✅ Strong Evidence Hypothesis is Correct
1. **Phase 1**: Improved prompts achieve ≥5% separation gap
2. **Steering**: Pos→neg achieves ≥-10pp lift (95% CI excludes 0)
3. **Steering**: Neg→pos achieves ≥+10pp lift (95% CI excludes 0)
4. **Specificity**: Neutral drift ≤3pp (operators are targeted)
5. **Operator Quality**: Fidelity ≥0.95 (learning converged)

### ⚠️ Partial Evidence
- Improved separation but weak steering → Blend ratio too low
- Strong steering but high neutral drift → Operators overfit
- One direction works, other doesn't → Asymmetric representation

### ❌ Hypothesis Wrong
- No improvement in separation → Prompts weren't the issue
- High separation but still null steering → Quantum→text disconnect
- Random operators perform equally well → Operators not learning

## Key Metrics to Report

### Phase 1: Representation Quality
- **Separation gap**: (within_class_fidelity - cross_class_fidelity)
- **Target**: ≥5% for good quantum representations
- **Old result**: ~4.8% (Qwen2.5-3B with biased prompts)

### Phase 2: Operator Quality
- **Fidelity**: |⟨ψ_target|U|ψ_source⟩|²
- **Target**: ≥0.95 (operators learned the transformation)
- **Old result**: 94% (operators trained correctly)

### Phase 3: Steering Effectiveness
- **Lift vs baseline**: Δ positive rate (percentage points)
- **Target**: ≥±10pp with 95% CI excluding 0
- **Old result**: ±0-2pp (no steering effect)

### Specificity: Neutral Drift
- **Metric**: |baseline_neutral_pos_rate - steered_neutral_pos_rate|
- **Target**: ≤3pp (operators don't affect neutral prompts)
- **Purpose**: Ensures operators are sentiment-specific, not just adding noise

## Files Created

### New Prompt Sets
- `prompts/improved_prompts_100.json` (50 pos + 50 neg neutral-stem prompts)
- `prompts/neutral_prompts_40.json` (40 truly neutral prompts for drift)
- `prompts/PROMPT_AUDIT_REPORT.md` (detailed analysis of original issues)

### Code Changes
- **decode_method threading**: Can now test `real_component` vs `real_imag_avg`
- **neutral_prompts support**: Evaluate specificity automatically

### Study Configs
- `experiments/sentiment/study_configs/improved_prompts_validation.json`

## Timeline

**Tonight**:
- Phase 1 collection: ~15-20 min
- Phase 2 training: ~10-15 min
- Phase 3 evaluation: ~30-40 min per blend ratio
- **Total per experiment**: ~1-2 hours

**Tomorrow Morning**:
- Pull results from GPU
- Analyze metrics
- Prepare summary for mentor

## What to Tell Mentor Tomorrow

### If Experiments Pass ✅
"**The null steering was caused by train/test distribution mismatch.** Operators trained on sentiment-biased prompts ('absolutely incredible and') couldn't generalize to neutral test prompts ('The weather today'). After switching to neutral-stem prompts that allow authentic continuations, we achieved [X]pp steering lift with high statistical significance and minimal neutral drift."

### If Experiments Partially Pass ⚠️
"**We've isolated the issue to [blend ratio | decode method | model size].** Improved prompts restored quantum separation to [X]%, but steering required [higher blend ratios | different decoding | larger models]. Next steps: [specific action]."

### If Experiments Fail ❌
"**The null steering persists despite fixing prompts.** This rules out distribution mismatch and points to a fundamental quantum→text disconnect. Possible causes: [off-manifold activations | inadequate blend strength | architecture-specific issues]. Recommend: [classical baseline | activation analysis | different intervention method]."

## Background: What We Learned Before

### Null Steering Observations (Nov 4-5)
1. **Pythia-410M (Layer 22)**: 100 prompts, no steering (~0pp lift)
2. **Qwen2.5-3B (Layer 31)**: Original prompts, minimal steering (~2pp lift)
3. **Both had**:
   - High operator fidelity (94%)
   - Decent separation gap (4.8%)
   - But zero behavioral effect

### Diagnostic Tests Revealed
1. **Asymmetric steering**: U_pos→neg increased positive sentiment (wrong direction!)
2. **Prompt audit**: Original prompts force sentiment ("absolutely incredible and")
3. **Train/test mismatch**: Models never saw neutral completions during training

## Commands for Tomorrow Morning

### Pull Results
```bash
rsync -avz -e "ssh -p 48513" \
  root@149.7.4.151:/workspace/cogit-qmech/results/ \
  ~/cogit-qmech-backup/results/
```

### Analyze
```python
import json
with open('results/quantum_intervention/evaluation_qwen_local_50_[timestamp].json') as f:
    results = json.load(f)

summary = results['summary']
print(f"Pos→neg lift: {summary['pos_to_neg']['lift_vs_baseline']:.1f}pp")
print(f"95% CI: {summary['pos_to_neg']['lift_ci_vs_baseline']}")
print(f"Neutral drift: {abs(summary['pos_to_neg']['positive_rate'] - summary['baseline']['positive_rate']):.1f}pp")
```

## Notes
- All experiments use same random seed (42) for reproducibility
- Operators saved with fingerprint-based naming (no collision)
- Logs saved with descriptive names for easy tracking
- GPU can handle multiple Phase 1 collections in parallel (model loading dominates)

---
**Status**: Experiment 1 launched successfully
**Next**: Monitor overnight, analyze tomorrow morning
**ETA for all results**: ~6-8 hours (depends on GPU availability)
