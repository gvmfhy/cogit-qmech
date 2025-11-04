# Transfer Attack Research Protocol

**Project**: Cogit-QMech
**Research Question**: Are quantum-encoded adversarial states universal across model scales?
**Authors**: Austin Morrissey & Claude Sonnet 4.5
**Date**: 2025-11-04

---

## Executive Summary

This protocol defines the experimental methodology for testing whether unitary operators trained on small models (Qwen2.5-7B) can manipulate large models (Qwen2.5-70B) using the same quantum-encoded states.

**Why this matters**: If transfer succeeds → quantum adversarial attacks are model-agnostic (security risk). If transfer fails → model-specific defenses are possible (safety advantage).

---

## Research Questions

### Primary Question
**Do quantum-encoded cognitive manipulations transfer across model scales?**

Specifically:
- Can unitary operators `U_pos→neg` trained on 7B successfully shift 70B sentiment?
- Do quantum states from 7B map meaningfully to 70B's representation space?
- Is the quantum structure model-agnostic or model-specific?

### Secondary Questions
1. **Scaling behavior**: If transfer occurs, does effectiveness scale linearly with model size?
2. **Layer depth**: Do attacks transfer better at early layers (low-level features) vs late layers (high-level semantics)?
3. **Architectural invariance**: Do attacks transfer between different architectures (e.g., Qwen → Llama)?
4. **Defensive strategies**: If transfer succeeds, what interventions can block it?

---

## Methodology

### Phase 1: Source Model Training (7B)

**Goal**: Train quantum operators on Qwen2.5-7B as "attack source"

**Steps**:
1. Run full Phase 1-4 pipeline on Qwen2.5-7B (qwen_remote preset)
2. Validate that operators achieve:
   - Training fidelity > 0.85
   - Reversibility > 0.85
   - Sentiment shift observable at α = 0.05-0.10
3. **Save critical artifacts**:
   - `encoder_projection_qwen2.5-7B_latest.pt` (quantum encoder)
   - `unitary_pos_to_neg_qwen2.5-7B_latest.pt` (trained operator)
   - `unitary_neg_to_pos_qwen2.5-7B_latest.pt` (trained operator)

**Hardware**: RunPod RTX 5090 (32GB) - $0.89/hr
**Expected time**: 15-20 min
**Cost**: ~$0.30

---

### Phase 2: Target Model Setup (70B)

**Goal**: Deploy and test Qwen2.5-70B for transfer attack testing

**Steps**:
1. **Deploy A100 80GB instance**:
   - RunPod A100 80GB (~$1.50-2.00/hr)
   - Minimum 80GB VRAM needed for 70B inference
   - Estimated 20GB for model + 10GB for activations + overhead
2. **Clone repo and setup**:
   - SSH into A100 instance
   - `git clone` and install dependencies
   - Load Qwen2.5-70B with TransformerLens
3. **Validate 70B baseline**:
   - Confirm model loads successfully
   - Extract sample activations
   - Verify CUDA/memory usage is sustainable

**Hardware**: RunPod A100 80GB
**Expected time**: 10 min setup + validation
**Cost**: ~$0.30 for setup

---

### Phase 3: Transfer Attack Experiment

**Goal**: Test if 7B-trained operators manipulate 70B representations

#### 3.1: Matched Encoding Strategy

**Approach**: Use **same projection matrix** from 7B encoder to encode 70B activations

**Rationale**: If quantum structure is universal, the same projection should work across models.

**Steps**:
1. Load `encoder_projection_qwen2.5-7B_latest.pt` (from Phase 1)
2. Extract 70B activations from **matching relative layer depth**:
   - 7B uses layer 14/28 (50% depth)
   - 70B has 80 layers → use layer 40 (50% depth)
3. Encode 70B activations using 7B projection:
   ```python
   encoder_7B = QuantumStateEncoder.load_projection(projection_file_7B)
   quantum_state_70B = encoder_7B.encode(activation_70B)
   ```
4. Apply 7B-trained operator:
   ```python
   U_7B = load_unitary_operator(operator_file_7B)
   transformed_70B = U_7B(quantum_state_70B)
   ```
5. Decode and inject back into 70B:
   ```python
   activation_modified = decoder.decode(transformed_70B)
   inject_into_70B(activation_modified, layer=40)
   ```
6. Measure sentiment shift on neutral prompts

#### 3.2: Independent Encoding Strategy (Control)

**Approach**: Train **new encoder + operators** directly on 70B

**Rationale**: Baseline to compare against transfer attack. If 70B-native operators work but 7B transfers don't, structure is model-specific.

**Steps**:
1. Run Phase 1-2 on 70B natively (collect 70B states, train 70B operators)
2. Test 70B-native operators on 70B (should work well)
3. Compare effectiveness: 70B-native vs 7B-transferred

---

### Phase 4: Transfer Effectiveness Analysis

**Metrics to measure**:

1. **Direct transfer success** (7B operator → 70B model):
   - Sentiment shift magnitude at α = 0.05, 0.10, 0.20
   - Coherence preservation (perplexity, grammaticality)
   - Behavioral change (positive → negative sentiment in completions)

2. **Comparison to native performance**:
   - Transfer effectiveness ratio: `(7B→70B shift) / (70B→70B shift)`
   - If ratio ≈ 1.0 → perfect transfer (universal attack)
   - If ratio ≈ 0.0 → no transfer (model-specific)
   - If ratio = 0.3-0.7 → partial transfer (scaling needed)

3. **Layer depth analysis**:
   - Test transfer at multiple layer depths (25%, 50%, 75%)
   - Hypothesis: Earlier layers (low-level features) transfer better

4. **Architectural invariance** (future work):
   - Test Qwen2.5-7B operator → Llama-3-8B
   - Test Qwen2.5-7B operator → Mistral-7B
   - Measure cross-architecture transfer

---

## Success Criteria

### Strong Transfer (Universal Attack)
- 7B operator achieves **>80%** of 70B-native operator's sentiment shift
- Coherence remains high (perplexity increase <20%)
- Transfer works across multiple layer depths

**Interpretation**: Quantum-encoded attacks are model-agnostic → **security risk**. Adversarial states crafted on small models compromise large models.

### Weak Transfer (Partial Scaling)
- 7B operator achieves **30-80%** of 70B-native effectiveness
- Scaling factor needed but transfer still observable

**Interpretation**: Quantum structure has some universality but requires calibration per model scale.

### No Transfer (Model-Specific)
- 7B operator achieves **<30%** of 70B-native effectiveness
- No meaningful sentiment shift at any blend ratio

**Interpretation**: Quantum structure is model-specific → **defensive advantage**. Each model has unique quantum geometry that resists external manipulation.

---

## Expected Outcomes & Hypotheses

### Hypothesis 1: Strong Transfer (Likely)
**Prediction**: Transfer will succeed because:
- Both models are Qwen2.5 family (same architecture, same pretraining)
- Linear representations hypothesis suggests shared structure
- Quantum encoding captures universal geometric features

**Evidence needed**:
- Transfer ratio > 0.8
- Works at multiple layer depths
- Coherence preserved

### Hypothesis 2: Scaling Effect (Moderate Likelihood)
**Prediction**: Transfer will partially succeed with scaling:
- 70B has richer representations (3840-d vs 3584-d hidden)
- Blend ratio α needs adjustment (e.g., α_70B = 0.5 * α_7B)
- Effect is weaker but observable

**Evidence needed**:
- Transfer ratio 0.3-0.8
- Scaling factor discoverable
- Still concerns for security

### Hypothesis 3: No Transfer (Unlikely)
**Prediction**: Transfer will fail because:
- Model-specific features dominate (emergent properties at scale)
- Quantum projection is model-dependent
- 70B has fundamentally different geometry

**Evidence needed**:
- Transfer ratio < 0.3
- No blend ratio achieves meaningful shift
- Layer depth doesn't matter

---

## Cost & Time Estimates

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Phase 1: 7B training | RTX 5090 | 20 min | $0.30 |
| Phase 2: 70B setup | A100 80GB | 10 min | $0.30 |
| Phase 3.1: Transfer test | A100 80GB | 30 min | $0.75 |
| Phase 3.2: Native baseline | A100 80GB | 30 min | $0.75 |
| Phase 4: Analysis | A100 80GB | 20 min | $0.50 |
| **Total** | | **110 min** | **$2.60** |

**RunPod credits available**: $7.48
**Remaining after transfer tests**: ~$4.88 (enough for additional experiments)

---

## Implementation Notes

### Code Modifications Needed

1. **Create `experiments/transfer/transfer_attack.py`**:
   - Load 7B projection matrix
   - Load 7B unitary operators
   - Extract 70B activations at matched layer depth
   - Apply 7B operators to 70B states
   - Measure transfer effectiveness

2. **Add 70B config preset** (`config.py`):
   ```python
   @classmethod
   def qwen_70b_remote(cls) -> 'QuantumConfig':
       return cls(
           model_name="Qwen/Qwen2.5-70B-Instruct",
           input_dim=3840,  # 70B hidden dim (confirm this)
           quantum_dim=10000,  # Keep same quantum dim as 7B for transfer
           target_layer=40,  # 50% depth (80 layers total)
           batch_size=8,  # Smaller batch due to 70B size
           epochs=100,
           device='cuda'
       )
   ```

3. **Modify `model_adapter_tl.py`** (if needed):
   - Ensure TransformerLens supports Qwen2.5-70B
   - May need to adjust for larger model (distributed loading, etc.)

---

## Defensive Implications

### If Transfer Succeeds → Security Research Needed
- **Detection**: Can we detect quantum-encoded attacks?
- **Mitigation**: Add noise to activations? Projection invariance?
- **Robustness**: Train models to be quantum-manipulation-resistant?

### If Transfer Fails → Safety Advantage
- **Model-specific defenses**: Each model has unique quantum signature
- **Adversarial robustness**: Quantum operators don't generalize
- **Red-teaming**: Still need to test each model individually

---

## Next Steps

1. ✅ Complete Phase 1: 7B baseline (tonight)
2. 📅 Deploy A100 80GB instance (tomorrow)
3. 🔬 Run transfer attack experiment
4. 📊 Analyze results and document findings
5. 📝 Write up results for potential publication/blog post

---

## References & Related Work

- **Linear representation hypothesis**: Suggests neural networks use linear geometry for concepts
- **Adversarial transferability**: Prior work shows adversarial examples often transfer across models
- **Unitary networks**: Literature on unitary RNNs for stability
- **Quantum ML**: Quantum-inspired ML techniques

---

## Appendix: Alternative Experiments

### Experiment A: Cross-Architecture Transfer
- Test Qwen2.5-7B operator → Llama-3-8B
- Requires implementing Llama support in model_adapter_tl.py

### Experiment B: Multi-Layer Transfer
- Train operators at multiple layers (early, mid, late)
- Test if early-layer operators transfer better (low-level features)

### Experiment C: Adversarial Robustness
- Can we train 70B to resist 7B-transferred attacks?
- Adversarial training in quantum space

---

**Status**: Ready to execute Phase 1 (tonight), then proceed to Phase 2-4
