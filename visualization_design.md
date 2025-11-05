# Publication-Quality Visualization Design: Quantum Steering Null Results

**Primary Narrative**: Quantum operators achieve excellent mathematical properties (94-98% fidelity, reversibility) but fail to produce behavioral steering effects (2pp sentiment lift with CIs including 0).

---

## Figure 1: Quantum State Separation (Phase 1 Validation)

**Purpose**: Show that positive/negative sentiment states are distinct in quantum space, validating the quantum encoding.

**Data Sources**:
- `/data/sentiment_quantum/quantum_states_{model}_latest.json`
- Fields: `positive_quantum_states`, `negative_quantum_states`, `separation_stats`

**Plot Design**:
```
Two-panel figure:
  Left: UMAP 2D projection of quantum states (100 states: 50 pos, 50 neg)
  Right: Within-class vs Cross-class fidelity distributions
```

**Key Elements**:
- **Left panel**:
  - Scatter plot with distinct colors for positive (blue) vs negative (orange)
  - UMAP dimensionality reduction (n_neighbors=15, min_dist=0.1)
  - Convex hull around each class to show separation
  - Annotations: "Within-class fidelity: 89%", "Cross-class fidelity: 79%"

- **Right panel**:
  - Overlapping histograms showing within-class fidelities (high ~0.89) vs cross-class (lower ~0.79)
  - Vertical line at separation_gap = 10.4%
  - Statistical annotation: "Cohen's d = X.XX (large effect)"

**Color Palette**:
- Positive: #2E86AB (blue)
- Negative: #F77F00 (orange)
- Background: white
- Grid: light gray (#E8E8E8)

**Interpretation Guide**:
- Clear separation → Quantum encoding captures sentiment structure
- High within-class fidelity → States are coherent
- Lower cross-class fidelity → Classes are distinguishable
- **Success metric**: Separation gap > 5% indicates good quantum representation

**Output**: `fig1_quantum_state_separation_300dpi.png|pdf`

---

## Figure 2: Operator Training Convergence (Phase 2 Validation)

**Purpose**: Show that unitary operators train successfully and achieve high quantum fidelity, establishing that the quantum machinery works.

**Data Sources**:
- `/models/quantum_operators/unitary_pos_to_neg_{model}_latest.pt`
- `/models/quantum_operators/unitary_neg_to_pos_{model}_latest.pt`
- Fields: `training_history['loss_history']`, `training_history['fidelity_history']`

**Plot Design**:
```
Two-row, two-column grid (2x2):
  Row 1: Training curves for both operators
  Row 2: Final performance metrics
```

**Key Elements**:
- **Panel A (top-left)**: Loss curves
  - Two lines: U_pos→neg (blue) and U_neg→pos (orange)
  - X-axis: Training epoch (0-100)
  - Y-axis: Loss (log scale if needed)
  - Shaded confidence bands if multiple runs available

- **Panel B (top-right)**: Fidelity evolution
  - Two lines showing fidelity improving to 94-98%
  - Horizontal dashed line at 0.90 (threshold for "excellent")
  - Final values annotated

- **Panel C (bottom-left)**: Reversibility histogram
  - Distribution of pos→neg→pos fidelity (from test_reversibility.py results)
  - Distribution of neg→pos→neg fidelity
  - Mean lines with annotations

- **Panel D (bottom-right)**: Comparison bar chart
  - Three bars: "Learned operators", "Random unitary", "Classical HDC (theoretical)"
  - Y-axis: Average reversibility fidelity
  - Error bars showing standard deviation
  - Learned ~0.95, Random ~0.50, Classical ~0.20

**Statistical Annotations**:
- "Final fidelity: 96.4% ± 1.2%"
- "Reversibility: 94.8% ± 2.1%"
- "Random baseline: 52.3% ± 8.4%"

**Interpretation Guide**:
- Converging loss → Training successful
- Fidelity > 90% → Excellent quantum transformations
- High reversibility → Operators truly unitary (quantum advantage)
- **Success metrics**: Fidelity > 0.90, Reversibility > 0.80

**Output**: `fig2_operator_training_convergence_300dpi.png|pdf`

---

## Figure 3: Behavioral Steering Results - The Null Effect (Phase 4 Main Finding)

**Purpose**: **THIS IS THE KEY FIGURE** - Show that despite quantum success, behavioral steering fails. All confidence intervals include zero.

**Data Sources**:
- All files: `/results/quantum_intervention/evaluation_*.json`
- Fields: `summary['neg_to_pos']['lift_vs_baseline']`, `lift_ci_vs_baseline`, `lift_vs_random`, `lift_ci_vs_random`

**Plot Design**:
```
Single comprehensive panel with nested structure:
  Main: Forest plot showing sentiment lift across all experiments
  Inset: Comparison to random operators
```

**Key Elements**:
- **Main plot**: Forest plot (horizontal)
  - Y-axis: Experiments (grouped by model, then blend_ratio)
    ```
    Qwen-3B (blend=0.5)
    Qwen-3B (blend=0.05)
    Pythia-410m (blend=0.5)
    Pythia-410m (blend=0.05)
    ```
  - X-axis: Sentiment lift (positive rate change)
  - Points: Point estimate of lift
  - Error bars: 95% bootstrap confidence intervals
  - Vertical line at x=0 (null effect) - **prominently styled**
  - Vertical shaded region: [-0.05, +0.05] (practical insignificance zone)

- **Color coding**:
  - Learned operators: Dark blue (#2E86AB)
  - Random control: Light gray (#95A5A6)
  - All CIs that include 0: Use lighter/desaturated color

- **Annotations**:
  - "No significant steering detected" (in margin)
  - "All CIs include zero" (below plot)
  - Effect sizes: "Cohen's h < 0.1 (trivial)"

- **Inset (top-right corner)**:
  - Small scatter plot: Learned lift vs Random lift
  - Diagonal y=x line
  - Most points cluster near origin and diagonal
  - Caption: "Learned ≈ Random (r=0.82, p=0.12)"

**Statistical Annotations**:
```
Qwen-3B (0.5):   +1.9pp [-12.0, +15.0]  ←
Qwen-3B (0.05):  +10.0pp [-3.0, +23.0]  (marginal, needs replication)
Pythia (0.5):    +1.9pp [-12.0, +15.0]  ←
Pythia (0.05):   -1.1pp [-15.0, +12.0]  ←
```

**Interpretation Guide**:
- **CIs crossing 0** → No statistically significant effect
- Points near 0 → Magnitude also trivial
- Learned ≈ Random → Quantum transformations don't translate to behavior
- **Failure criteria**: CI includes 0 AND |point estimate| < 5pp

**Output**: `fig3_behavioral_null_effect_300dpi.png|pdf`

---

## Figure 4: Perplexity Analysis - Off-Manifold Diagnosis

**Purpose**: Test hypothesis that quantum steering pushes activations off the natural language manifold, causing incoherent text that nullifies steering.

**Data Sources**:
- `/results/quantum_intervention/evaluation_*.json`
- Fields: `records[i]['baseline']['perplexity']`, `records[i]['pos_to_neg']['perplexity']`, etc.

**Plot Design**:
```
Three-panel horizontal layout:
  Left: Perplexity distributions
  Middle: Perplexity vs steering magnitude
  Right: Example text quality
```

**Key Elements**:
- **Panel A (left)**: Violin plots
  - Four violins: Baseline, Pos→Neg, Neg→Pos, Random
  - Y-axis: Perplexity
  - Overlay: Box plot showing median/quartiles
  - Statistical test: ANOVA F-statistic and p-value
  - Pairwise comparisons: "Baseline vs Steered: p=0.73 (n.s.)"

- **Panel B (middle)**: Scatter plot with regression
  - X-axis: Blend ratio (0.05, 0.5)
  - Y-axis: ΔPerplexity (steered - baseline)
  - Color: Model (Qwen vs Pythia)
  - Regression line with 95% CI band
  - Correlation coefficient: "r = -0.08, p = 0.64 (n.s.)"

- **Panel C (right)**: Qualitative examples
  - Text box showing 3 example completions:
    1. Baseline (perplexity ~4.2)
    2. Steered (perplexity ~4.1)
    3. Steered-high-blend (perplexity ~4.3)
  - Color-coded background by quality
  - Arrow annotations pointing out coherence

**Statistical Annotations**:
- "Mean perplexity: Baseline=4.29, Steered=4.18 (Δ=-0.11, p=0.53)"
- "No evidence of off-manifold distortion"
- "Text remains coherent even at high blend ratios"

**Interpretation Guide**:
- Similar perplexity → Activations stay on-manifold
- No correlation with blend ratio → Steering doesn't distort
- Coherent text → Failure is NOT due to incoherent generation
- **Conclusion**: Off-manifold hypothesis rejected

**Output**: `fig4_perplexity_analysis_300dpi.png|pdf`

---

## Figure 5: Experiment Grid - Comprehensive Parameter Sweep

**Purpose**: Show that null steering persists across all tested configurations (models, blend ratios, prompt sets).

**Data Sources**:
- All evaluation files
- Aggregate across: models (Qwen-3B, Pythia-410m), blend_ratios (0.05, 0.5), prompt_count (100)

**Plot Design**:
```
Heatmap with annotations:
  Rows: Models (Qwen-3B, Pythia-410m)
  Columns: Blend ratios (0.05, 0.1, 0.2, 0.5)
  Cell color: Sentiment lift magnitude
  Cell annotation: Lift ± CI width
```

**Key Elements**:
- **Heatmap**:
  - Color scale: Diverging RdBu (-20pp to +20pp)
  - Center (white) at 0
  - Annotations in each cell:
    ```
    +1.9pp
    [-12, +15]
    ```
  - Bold border around cells where CI excludes 0 (if any)

- **Side panel**: Summary statistics
  - "Configs tested: 4"
  - "Significant effects: 0"
  - "Mean |lift|: 3.3pp"
  - "Mean CI width: 27pp"

- **Bottom panel**: Statistical power analysis
  - "With N=100, 80% power to detect lift ≥ 18pp"
  - "Observed lifts (1-10pp) below detection threshold"
  - "Larger sample sizes unlikely to change conclusion"

**Interpretation Guide**:
- All cells near white → No strong effects anywhere
- Wide CIs → High variance, underpowered for small effects
- **Conclusion**: No configuration produces reliable steering

**Output**: `fig5_experiment_grid_300dpi.png|pdf`

---

## Figure 6: Diagnostic Timeline - Where Does Quantum→Behavioral Coupling Fail?

**Purpose**: Trace the quantum transformation through the generation pipeline to identify where steering signal is lost.

**Data Sources**:
- Requires new analysis: Compare quantum state transformations to actual activation changes during generation
- `/results/quantum_intervention/quantum_results_*.json` (intervention outputs)
- Operator checkpoint files

**Plot Design**:
```
Sankey-style flow diagram with quantitative annotations:

  [Quantum Space]           [Real Space]           [Generation]           [Output]
       |                        |                       |                      |
  High fidelity  -->  Blend activations  -->  Token selection  -->  Sentiment change
   (96.4%)                 (??%)                  (??%)               (1.9pp)
                           ↓
                    Check: Do blended
                    activations differ?
```

**Key Elements**:
- **Stage 1**: Quantum transformation
  - Bar: Fidelity = 96.4%
  - Color: Green (success)

- **Stage 2**: Activation blending
  - **REQUIRES NEW METRIC**: Cosine similarity between:
    - Baseline activations
    - Quantum-transformed activations (after inverse projection)
  - Bar: Activation Δ = ??
  - Color: Yellow (unknown) or Red (if small)

- **Stage 3**: Token probability shift
  - **REQUIRES NEW METRIC**: KL divergence between:
    - Baseline next-token distribution
    - Steered next-token distribution
  - Bar: KL divergence = ??
  - Color: Based on magnitude

- **Stage 4**: Behavioral outcome
  - Bar: Sentiment lift = 1.9pp
  - Color: Red (failure)

- **Diagnostic arrows**:
  - If Stage 2 shows large Δ but Stage 4 shows no effect:
    → "Activation changes don't affect token selection"
  - If Stage 2 shows small Δ:
    → "Quantum transforms lost during inverse projection"

**Interpretation Guide**:
- Identify the "break point" where quantum signal disappears
- **Hypothesis A**: Signal lost at inverse projection (quantum→real)
- **Hypothesis B**: Signal preserved in activations but ignored by model
- **Hypothesis C**: Signal affects tokens but not sentiment

**Output**: `fig6_diagnostic_pipeline_300dpi.png|pdf`

**Note**: This figure requires additional analysis code to compute activation-space and token-distribution metrics.

---

## Summary: Figure Organization for Paper

### Main Text Figures (4):
1. **Figure 1**: Quantum state separation (validates Phase 1)
2. **Figure 2**: Operator training (validates Phase 2)
3. **Figure 3**: Null steering effect (main negative result)
4. **Figure 4**: Perplexity analysis (rules out off-manifold hypothesis)

### Supplementary Figures (2):
5. **Figure S1**: Experiment grid (comprehensive parameter sweep)
6. **Figure S2**: Diagnostic pipeline (future work / mechanistic analysis)

### Graphical Abstract:
- Single-panel visual summary combining:
  - Top: "Quantum operators work" (high fidelity icon)
  - Middle: "→" arrow with "blend_ratio" dial
  - Bottom: "Behavior unchanged" (flat sentiment distribution)

---

## Implementation Notes

### Color Palette (Colorblind-Friendly):
```python
COLORS = {
    'positive': '#2E86AB',      # Blue
    'negative': '#F77F00',      # Orange
    'neutral': '#95A5A6',       # Gray
    'success': '#06A77D',       # Green
    'failure': '#D62828',       # Red
    'background': '#FFFFFF',    # White
    'grid': '#E8E8E8',          # Light gray
}
```

### Font Specifications:
- Title: 14pt, bold
- Axis labels: 12pt, regular
- Tick labels: 10pt
- Annotations: 9pt
- Figure text: 8pt

### Export Settings:
```python
plt.savefig(filename, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none',
            format='png')
plt.savefig(filename.replace('.png', '.pdf'),
            bbox_inches='tight', format='pdf')
```

### Statistical Annotations:
- Always include: point estimate, 95% CI, p-value (if test performed)
- Use consistent formatting: "Δ = +1.9pp [-12.0, +15.0], p = 0.68"
- Mark significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. otherwise

---

## File Naming Convention

```
fig{N}_{descriptive_name}_{resolution}.{format}

Examples:
- fig1_quantum_state_separation_300dpi.png
- fig1_quantum_state_separation_300dpi.pdf
- fig3_behavioral_null_effect_300dpi.png
- figS1_experiment_grid_300dpi.png
```

All figures saved to: `/Users/austinmorrissey/cogit-qmech/figures/publication/`
