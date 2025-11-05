# Cogit-QMech: Quantum Cognitive Operators

A quantum-mechanically rigorous framework for manipulating cognitive states in neural networks using **unitary transformations** and **Born rule probabilities**.

## Motivation

This project extends the classical HDC approach from [project-cogit-framework](../project-cogit-framework) by rebuilding the cognitive manipulation framework on quantum mechanical foundations. The classical version, while effective, was functionally similar to standard steering techniques due to the lack of **unitary constraints** - transformations were not reversible and didn't preserve the mathematical structure of quantum operations.

## Key Differences from Classical Version

| Feature | Classical (HDC) | Quantum (This Repo) |
|---------|----------------|---------------------|
| **State representation** | Binary vectors {-1, +1} | Complex amplitudes (normalized) |
| **Operator type** | MLP (arbitrary transforms) | Unitary (U†U = I) |
| **Loss function** | MSE (cosine similarity) | Born rule fidelity |
| **Reversibility** | ❌ Not reversible | ✅ Reversible (round-trip) |
| **Distance metric** | Hamming / Cosine | Quantum fidelity \|⟨ψ\|φ⟩\|² |

## Quantum Foundations

### Complex-Valued Quantum States
Neural network activations are encoded as **normalized complex-valued vectors**:
- Real activation (768-d) → Quantum state \|ψ⟩ (10,000-d complex)
- Normalization: ⟨ψ\|ψ⟩ = 1
- Preserves information via complex amplitudes + phases

### Unitary Operators
Transformations are **unitary matrices** satisfying U†U = I:
- Implemented via Cayley transform: U = (I + iH)(I - iH)⁻¹
- Preserves state norms: \|U\|ψ⟩\| = \|\|ψ⟩\|
- **Reversible**: U† undoes the transformation

### Born Rule Probabilities
Training uses quantum fidelity instead of classical similarity:
- Loss: L = 1 - \|⟨target\|U\|source⟩\|²
- Maximizes quantum probability of measuring target state
- Aligns with quantum measurement theory

## Research Questions

1. **Do neural network activations exhibit quantum-like structure?**
   - Can we find meaningful unitary transformations?
   - Does unitarity improve intervention quality?

2. **Is reversibility achievable?**
   - Can we map positive→negative→positive with high fidelity?
   - Does this preserve semantic meaning better than classical approaches?

3. **Performance vs. classical HDC:**
   - Stronger sentiment shifts at lower blend ratios?
   - Better text coherence during intervention?

## Important: Layer Selection

**⚠️ Layer selection is critical for quantum steering success.**

Before training on a new model, always run a layer sweep to find optimal sentiment separation. See **[docs/LAYER_SELECTION.md](docs/LAYER_SELECTION.md)** for detailed analysis showing:

- **Wrong layer = zero steering** regardless of operator quality
- Pythia-410M layer 12: 0.40% separation → steering failed
- Pythia-410M layer 22: 5.41% separation → 13.5x improvement
- Sentiment separation emerges in late layers (85-95% depth), not middle layers

The separation gap metric (`within-class fidelity - cross-class fidelity`) quantitatively predicts steering success before training.

## Repository Structure

```
cogit-qmech/
├── src/
│   ├── quantum_encoder.py      # Real → Complex quantum states
│   ├── unitary_operator.py     # U†U = I constrained neural network
│   ├── quantum_decoder.py      # Complex → Real with measurement
│   ├── quantum_utils.py        # Fidelity, inner products, unitarity checks
│   └── model_adapter_tl.py     # TransformerLens adapter (from classical)
├── experiments/
│   └── sentiment/
│       ├── quantum_phase1_collect.py    # Collect complex cogits (with layer sweep)
│       ├── quantum_phase2_train.py      # Train unitary operators
│       ├── quantum_phase3_test.py       # Test interventions
│       └── test_reversibility.py        # Validate round-trip fidelity
├── docs/
│   └── LAYER_SELECTION.md               # Critical guide for choosing optimal layers
├── analysis/
│   ├── quantum_vs_classical.py          # Compare frameworks
│   └── visualize_quantum.py             # Plot complex amplitudes
└── tests/                               # Unit tests for quantum ops
```

## Installation

```bash
# Clone the repository
cd ~/
git clone [repository-url] cogit-qmech
cd cogit-qmech

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install quantum ML libraries
pip install -r requirements.quantum.txt
```

## Quick Start

### Phase 1: Collect Quantum States
```bash
python experiments/sentiment/quantum_phase1_collect.py
```
Encodes GPT-2 activations as complex-valued quantum states.

### Phase 2: Train Unitary Operators
```bash
python experiments/sentiment/quantum_phase2_train.py
```
Learns unitary transformations U_pos→neg and U_neg→pos.

### Phase 3: Test Interventions
```bash
python experiments/sentiment/quantum_phase3_test.py
```
Applies quantum operators to GPT-2 generation with gentle blending.

### Test Reversibility
```bash
python experiments/sentiment/test_reversibility.py
```
Validates round-trip fidelity: state → U → U† → state'

## Expected Outcomes

**Success would demonstrate:**
- ✅ **Reversibility**: F(state, round-trip) > 0.9
- ✅ **Efficiency**: Stronger shifts at lower blend ratios (e.g., 0.02 vs 0.1)
- ✅ **Coherence**: Better text quality during intervention
- ✅ **Theory**: Evidence for quantum-like structure in neural representations

**Even if quantum ≈ classical:**
- Rules out quantum effects in LLM representations
- Provides insights about high-dimensional embeddings
- Validates or refutes quantum interpretations of neural networks

## Research Team

**Princple Investigator:** Bryce-allen Bagley, 
**Mentee**: Austin Morrissey
**Implementation**: Claude Sonnet 4.5

## Related Work

- **Classical HDC version**: [../project-cogit-framework](../project-cogit-framework)
- **TransformerLens**: [Neel Nanda's interpretability library](https://github.com/neelnanda-io/TransformerLens)
- **Unitary RNNs**: Arjovsky et al., "Unitary Evolution Recurrent Neural Networks" (2016)
- **Quantum Machine Learning**: Biamonte et al., "Quantum machine learning" (2017)

## Citation

If you use this work, please cite both the quantum and classical versions:

```
@software{cogit_qmech_2025,
  author = {Morrissey, Austin,  Bagley, Bryce-Allen},
  title = {Cogit-QMech: Quantum Cognitive Operators for Neural Network Manipulation},
  year = {2025},
  note = {Extends project-cogit-framework with quantum mechanical foundations}
}
```

## License

[To be determined - discuss with Austin]
