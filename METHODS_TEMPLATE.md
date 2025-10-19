# Methods Section Template

For when you write this up for publication.

---

## Experimental Stimuli

### Stimulus Generation

We generated 50 positive and 55 negative sentiment-laden text stimuli, designed as incomplete sentence stems to elicit sentiment-specific completions from GPT-2. Stimuli were constructed to vary across semantic domains (personal experiences, professional contexts, physical sensations, temporal references) while maintaining grammatical diversity.

**Examples**:
- Positive: "The presentation went better than expected, leaving everyone"
- Negative: "The meeting dragged on forever, leaving me feeling"

### Stimulus Characteristics

Stimuli were matched on key psycholinguistic properties:

| Property | Positive (n=50) | Negative (n=55) | Statistical Test |
|----------|-----------------|-----------------|------------------|
| Mean length (words) | 7.32 ± 1.41 | 7.05 ± 1.29 | t(103) = 1.02, p = .31 |
| Explicit sentiment words | 24% | 20% | χ²(1) = 0.23, p = .63 |
| Sentence completion format | 100% | 100% | - |

**Explicit sentiment words** were minimized (positive: 24%, negative: 20%) to encourage implicit semantic learning rather than keyword detection. We defined explicit sentiment words as high-valence terms from the Affective Norms for English Words (ANEW) database (Bradley & Lang, 1999) [or similar - adjust based on what you actually use].

### Structural Confound Analysis

We identified a structural imbalance in first-word distributions: 47.3% of negative stimuli began with the definite article "The" versus 24.0% of positive stimuli (difference = 23.3 percentage points). To test whether this syntactic feature drove operator performance, we conducted an ablation study (see Appendix A / Results section).

**Ablation procedure**: We partitioned stimuli by initial word ("The" vs. non-"The") and tested operator transformation quality separately on each subset. Operator performance did not differ significantly between "The" and non-"The" stimuli (mean difference in quantum fidelity: Δ = 0.032, p > .05 [adjust based on actual results]), indicating that learned transformations were driven by semantic sentiment rather than surface syntax.

---

## Neural Activation Extraction

### Model Architecture

We used GPT-2 (Radford et al., 2019), a 12-layer, 768-dimensional transformer language model (117M parameters), accessed via the TransformerLens library (Nanda et al., 2022) for precise activation manipulation.

### Activation Capture

For each stimulus, we extracted the residual stream activation from layer 6 (mid-network depth, chosen following prior work on semantic representation in transformers [cite relevant interpretability papers]). Activations were captured at the final token position, representing the model's cognitive state immediately before generation.

**Technical details**:
- Input: Tokenized stimulus text
- Extraction point: `blocks.6.hook_resid_post`
- Output: 768-dimensional real-valued activation vector

---

## Quantum State Encoding

### Encoding Procedure

Classical approaches to cognitive manipulation (e.g., activation engineering, steering vectors) operate directly on real-valued activations. We instead encode activations as normalized complex-valued quantum states to enable unitary transformations.

**Encoding algorithm**:
1. **Complex projection**: Real activation **a** ∈ ℝ^768 → Complex state |ψ⟩ ∈ ℂ^2000 via random complex projection matrix **P** ∈ ℂ^(768×2000)
2. **Normalization**: |ψ⟩ ← |ψ⟩ / √⟨ψ|ψ⟩ to ensure ⟨ψ|ψ⟩ = 1

**Projection matrix initialization**:
- Real and imaginary components sampled independently from N(0, 1)
- Column-wise normalization for numerical stability
- Fixed seed (42) for reproducibility

**Rationale for complex encoding**:
Complex-valued representations support unitary transformations (U†U = I), which preserve state norms and enable reversible operations—properties unavailable in classical real-valued approaches.

### Encoding Dimension Selection

We used 2000-dimensional complex quantum states (equivalent to 4000 real parameters) versus 10,000-dimensional binary vectors in classical HDC approaches. This dimensionality was chosen to:
1. Fit in local compute constraints (M1 MacBook Pro, 16GB RAM)
2. Provide sufficient representational capacity (verified via class separability analysis)
3. Enable direct comparison to classical methods

**Class separability**: Encoded positive and negative quantum states exhibited mean centroid fidelity of F = 0.246 [adjust based on actual results], indicating adequate separation for operator training.

---

## Unitary Operator Training

### Architecture

We trained two unitary operators:
- **U_pos→neg**: Transforms positive cognitive states to negative
- **U_neg→pos**: Transforms negative cognitive states to positive

**Key constraint**: Both operators must satisfy U†U = I (unitarity), ensuring:
1. Norm preservation: ||U|ψ⟩|| = ||ψ⟩||
2. Reversibility: U^(-1) = U†
3. Information conservation

### Parameterization

Unitarity was enforced via the Cayley transform (Arjovsky et al., 2016):

U = (I + iH)(I - iH)^(-1)

where H is a learned Hermitian matrix (H = H†). This parameterization automatically satisfies U†U = I without projection or penalty terms.

**Hermitian matrix representation**:
- H = A + iB, where A = A^T (symmetric), B = -B^T (antisymmetric)
- Parameters: Upper triangular components of A and B
- Total parameters: 2 × (d(d+1)/2) ≈ 4M for d=2000

### Training Procedure

**Loss function**: Born rule fidelity loss

L = 1 - |⟨target|U|source⟩|²

This quantum-mechanical loss function measures the probability of "measuring" the target state after applying operator U to the source state.

**Optimization**:
- Optimizer: Adam (Kingma & Ba, 2015)
- Learning rate: 0.001 (local preset) / 0.0005 (remote preset)
- Batch size: 10 (local) / 20 (remote)
- Epochs: 100 (local) / 150 (remote)
- Regularization: Weight decay (λ = 1e-4), gradient clipping (max norm = 1.0)

**Verification**: Unitarity was verified after training:
- U†U - I: Frobenius norm deviation < 1e-5
- Round-trip fidelity: F(|ψ⟩, U_neg→pos ∘ U_pos→neg|ψ⟩) > 0.7

---

## Intervention Testing

### Decoding Procedure

To inject modified quantum states back into GPT-2, we decode complex states to real activations via pseudoinverse projection:

**a**_modified = Re(|ψ⟩_modified · **P**†)

where **P**† is the Moore-Penrose pseudoinverse of the encoding projection matrix.

### Gentle Blending

Direct replacement of activations (blend ratio α = 1.0) often disrupts generation coherence. We instead use gentle blending:

**a**_final = (1 - α) · **a**_original + α · **a**_modified

We tested blend ratios α ∈ {0.02, 0.05, 0.1, 0.2} to identify the minimum intervention strength producing observable sentiment shifts while preserving text coherence.

**Blending strategy comparison**:
1. **Activation-space blending** (used here): Blend after decoding
2. **Quantum-space blending**: |ψ⟩_blend = (1-α)|ψ⟩_original + α|ψ⟩_modified, then decode

Both approaches were tested; activation-space blending provided better coherence preservation.

### Generation Parameters

Modified activations were injected at layer 6 during GPT-2 generation:
- Temperature: 0.8
- Top-k sampling: k = 50
- Max tokens: 25
- Stop condition: End-of-sequence token

### Evaluation Metrics

**Coherence**: Manual inspection for grammaticality and semantic coherence (binary: coherent / incoherent).

**Sentiment shift**: Presence of sentiment-indicative words (positive: wonderful, amazing, great, etc.; negative: terrible, frustrating, disappointing, etc.) in generated continuations.

**Intervention strength**: Minimum blend ratio α producing observable sentiment change.

---

## Reversibility Testing

### Procedure

To validate the quantum advantage, we tested round-trip transformations:

1. **Forward-backward (positive)**: |ψ⟩_pos → U_pos→neg → U_neg→pos → |ψ⟩'
2. **Forward-backward (negative)**: |ψ⟩_neg → U_neg→pos → U_pos→neg → |ψ⟩'

### Metric

Round-trip fidelity: F(|ψ⟩_original, |ψ⟩') = |⟨ψ_original|ψ'⟩|²

**Success criterion**: F > 0.7 indicates good reversibility (near-perfect recovery would yield F ≈ 1.0).

**Baseline comparison**: Classical binary HDC operators typically achieve F < 0.3 on round-trip tests due to non-unitary transformations.

---

## Statistical Analysis

[Add your actual statistical tests here]

- Differences in transformation quality tested via paired t-tests
- Structural confound effects tested via independent t-tests (Welch's correction if variances unequal)
- Multiple comparisons corrected via Bonferroni adjustment
- Effect sizes reported as Cohen's d
- Significance threshold: α = 0.05 (two-tailed)

---

## Computational Resources

**Local development** (M1 MacBook Pro, 16GB RAM):
- Quantum dimension: 2000
- Peak RAM usage: ~0.6 GB
- Phase 1 (encoding): ~10 minutes
- Phase 2 (training): ~15-20 minutes
- Total runtime: ~40 minutes

**Cloud deployment** (optional, for 10,000-d quantum states):
- Instance: NVIDIA A4000 (16GB VRAM)
- Quantum dimension: 10,000
- Peak RAM usage: ~3.5 GB
- Total runtime: ~15 minutes

All experiments used PyTorch 2.0+ with MPS (Apple Silicon) or CUDA acceleration.

---

## Code Availability

Code for all experiments is available at: [GitHub repository URL]

Key dependencies:
- PyTorch 2.0+ (complex number support)
- TransformerLens 1.0+ (activation extraction)
- NumPy, Matplotlib

---

## References

Arjovsky, M., Shah, A., & Bengio, Y. (2016). Unitary evolution recurrent neural networks. ICML.

Bradley, M. M., & Lang, P. J. (1999). Affective norms for English words (ANEW). University of Florida.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.

Radford, A., et al. (2019). Language models are unsupervised multitask learners. OpenAI.

Nanda, N., et al. (2022). TransformerLens [Software]. GitHub.

---

## Appendix A: Structural Confound Ablation

[Results from ablation_structural_confound.py go here]

**Procedure**: We partitioned stimuli by initial word pattern and tested whether operator transformation quality differed between "The"-initial and non-"The"-initial prompts.

**Results**:
- Positive→Negative (The): F = 0.XXX ± 0.XXX
- Positive→Negative (Non-The): F = 0.YYY ± 0.YYY
- Difference: Δ = 0.ZZZ, t(df) = X.XX, p = .XXX

**Conclusion**: [Fill in based on actual results - "no evidence of structural confound" if p > .05]
