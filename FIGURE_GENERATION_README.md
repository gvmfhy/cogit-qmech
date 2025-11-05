# Publication Figure Generation - Quick Start

This guide shows how to generate publication-quality figures for the quantum steering null results.

---

## Prerequisites

```bash
# Install required packages
pip install numpy matplotlib seaborn scipy scikit-learn umap-learn torch
```

---

## Quick Start

### Generate All Figures

```bash
cd /Users/austinmorrissey/cogit-qmech
python create_publication_figures.py --all
```

Output: 5 figures (10 files: PNG + PDF for each)
Location: `/Users/austinmorrissey/cogit-qmech/figures/publication/`

### Generate Specific Figures

```bash
# Just the main result (behavioral null effect)
python create_publication_figures.py --figure 3

# Multiple specific figures
python create_publication_figures.py --figure 1 2 3

# Individual figures
python create_publication_figures.py --figure 1  # Quantum separation
python create_publication_figures.py --figure 2  # Operator training
python create_publication_figures.py --figure 3  # Behavioral null (MAIN)
python create_publication_figures.py --figure 4  # Perplexity analysis
python create_publication_figures.py --figure 5  # Experiment grid
```

---

## Output Files

Each figure generates two files:

```
figures/publication/
├── fig1_quantum_state_separation_300dpi.png       (raster for web/presentations)
├── fig1_quantum_state_separation_300dpi.pdf       (vector for publication)
├── fig2_operator_training_convergence_300dpi.png
├── fig2_operator_training_convergence_300dpi.pdf
├── fig3_behavioral_null_effect_300dpi.png         ⭐ MAIN RESULT
├── fig3_behavioral_null_effect_300dpi.pdf
├── fig4_perplexity_analysis_300dpi.png
├── fig4_perplexity_analysis_300dpi.pdf
├── fig5_experiment_grid_300dpi.png
└── fig5_experiment_grid_300dpi.pdf
```

**File sizes**:
- PNG: ~500KB - 2MB (300 DPI, suitable for print)
- PDF: ~100KB - 500KB (vector, infinitely scalable)

---

## Figure Descriptions

### Figure 1: Quantum State Separation
**Purpose**: Validate that quantum encoding captures sentiment structure

**Panels**:
- Left: UMAP projection showing positive/negative clusters
- Right: Within-class vs cross-class fidelity distributions

**Key Result**: 10% separation gap confirms quantum encoding works

**Estimated runtime**: ~30 seconds (UMAP computation)

---

### Figure 2: Operator Training Convergence
**Purpose**: Validate that unitary operators train successfully

**Panels**:
- A: Loss curves for both operators
- B: Fidelity evolution over training
- C: Reversibility histogram (pos→neg→pos)
- D: Learned vs Random vs Classical comparison

**Key Result**: 96% fidelity and 95% reversibility = quantum operators work perfectly

**Estimated runtime**: ~10 seconds

---

### Figure 3: Behavioral Null Effect ⭐ MAIN RESULT
**Purpose**: Show that despite quantum success, behavioral steering fails

**Elements**:
- Main: Forest plot of sentiment lift across experiments
- Inset: Learned vs Random scatter plot

**Key Result**: All confidence intervals include 0 = no steering effect

**Estimated runtime**: ~15 seconds

**This is the figure for the abstract/graphical summary**

---

### Figure 4: Perplexity Analysis
**Purpose**: Rule out "off-manifold distortion" hypothesis

**Panels**:
- A: Violin plots comparing perplexity distributions
- B: Perplexity vs blend ratio scatter
- C: Example text quality comparison

**Key Result**: Similar perplexity (p>0.5) = text stays coherent, failure is not due to gibberish

**Estimated runtime**: ~20 seconds

---

### Figure 5: Experiment Grid
**Purpose**: Show null effect persists across all configurations

**Panels**:
- Left: Heatmap of sentiment lift (model × blend_ratio)
- Right: Summary statistics and power analysis

**Key Result**: No configuration works, effect sizes below detection threshold

**Estimated runtime**: ~10 seconds

---

## Customization

### Change Data Source

```bash
python create_publication_figures.py --data-root /path/to/cogit-qmech-backup
```

### Modify Figure Code

Edit `create_publication_figures.py`:

```python
# Example: Change color palette
COLORS = {
    'positive': '#YOUR_COLOR',
    'negative': '#YOUR_COLOR',
    # ...
}

# Example: Adjust figure size
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Larger

# Example: Change DPI
self.save_figure(fig, name)  # Uses 300 DPI by default
# Or manually:
fig.savefig('output.png', dpi=600)  # Higher resolution
```

---

## Troubleshooting

### Issue: "File not found" errors

**Cause**: Data files missing or in wrong location

**Fix**:
```bash
# Check data directory structure
ls -R /Users/austinmorrissey/cogit-qmech-backup/

# Expected structure:
# cogit-qmech-backup/
# ├── data/sentiment_quantum/
# │   ├── quantum_states_qwen2.5-3B_latest.json
# │   └── ...
# ├── models/quantum_operators/
# │   ├── unitary_pos_to_neg_qwen2.5-3B_latest.pt
# │   └── ...
# └── results/quantum_intervention/
#     ├── evaluation_*.json
#     └── ...

# If files are elsewhere, use --data-root:
python create_publication_figures.py --data-root /correct/path
```

### Issue: UMAP takes too long

**Cause**: Large quantum dimension (5333-D)

**Fix**: Reduce samples in code
```python
# In create_figure1_quantum_separation():
# Change from all samples to subset:
pos_states = pos_states[:50]  # First 50 only
neg_states = neg_states[:50]
```

### Issue: Figures look blurry

**Cause**: Using PNG at low DPI

**Fix**: Use PDF for publication
```bash
# PDFs are vector graphics (infinite zoom)
# Use the .pdf outputs for final paper submission
```

### Issue: Font rendering problems

**Cause**: Missing fonts on system

**Fix**:
```python
# In create_publication_figures.py, change font:
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],  # Available on most systems
})
```

---

## Advanced Usage

### Batch Processing

```bash
# Generate all figures with custom output directory
for i in 1 2 3 4 5; do
    python create_publication_figures.py --figure $i
done

# Or use GNU parallel for speed
parallel python create_publication_figures.py --figure ::: 1 2 3 4 5
```

### Integration with Paper

```latex
% In LaTeX paper:
\begin{figure}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/publication/fig3_behavioral_null_effect_300dpi.pdf}
    \caption{
        Behavioral steering null effect. Forest plot showing sentiment lift
        across all experiments. All 95\% confidence intervals include zero,
        indicating no significant steering effect despite perfect quantum
        operator performance (see Figure 2).
    }
    \label{fig:main_result}
\end{figure}
```

### Export for Presentations

```python
# Modify code for presentation slides (larger fonts):
plt.rcParams.update({
    'font.size': 14,           # Larger base font
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'figure.figsize': (12, 8), # Larger figures
})
```

---

## Quality Checklist

Before submitting figures:

- [ ] All text is readable at 100% zoom
- [ ] Color scheme is colorblind-friendly (use [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/))
- [ ] Statistical annotations are present (CIs, p-values)
- [ ] Axes are labeled with units
- [ ] Legends are clear and positioned well
- [ ] File sizes are reasonable (<5MB per figure)
- [ ] PDFs have embedded fonts
- [ ] Figures are referenced in text
- [ ] Captions explain key findings

---

## Data Sources Reference

### Figure 1 Data
```
Input:  data/sentiment_quantum/quantum_states_{model}_latest.json
Fields: positive_quantum_states, negative_quantum_states, separation_stats
Size:   ~20-80MB (complex vectors)
```

### Figure 2 Data
```
Input:  models/quantum_operators/unitary_pos_to_neg_{model}_latest.pt
        models/quantum_operators/unitary_neg_to_pos_{model}_latest.pt
Fields: training_history['loss_history'], training_history['fidelity_history']
Size:   ~20-700MB (unitary matrices)
```

### Figure 3 Data
```
Input:  results/quantum_intervention/evaluation_*.json (all files)
Fields: summary['neg_to_pos']['lift_vs_baseline'], 'lift_ci_vs_baseline'
Size:   ~50-400KB each (JSON results)
```

### Figure 4 Data
```
Input:  results/quantum_intervention/evaluation_*.json (all files)
Fields: records[i]['baseline']['perplexity'], records[i]['neg_to_pos']['perplexity']
Size:   ~50-400KB each (JSON results)
```

### Figure 5 Data
```
Input:  results/quantum_intervention/evaluation_*.json (all files)
Fields: config['model_name'], config['blend_ratio'], summary statistics
Size:   ~50-400KB each (JSON results)
```

---

## Expected Runtime

On a typical laptop (2023 MacBook Pro):

| Figure | Runtime  | Bottleneck               |
|--------|----------|--------------------------|
| 1      | ~30s     | UMAP projection          |
| 2      | ~10s     | PyTorch checkpoint load  |
| 3      | ~15s     | JSON parsing (many files)|
| 4      | ~20s     | JSON parsing + stats     |
| 5      | ~10s     | JSON parsing             |
| **Total** | **~90s** | **Full pipeline**        |

---

## Citation

When using these figures in publications:

```
Figures generated using custom visualization pipeline for quantum cognitive
steering experiments. Code available at: github.com/username/cogit-qmech

Figure design based on publication standards from Nature Communications
and PLOS ONE data visualization guidelines.
```

---

## Support

For issues or questions:

1. Check this README
2. Consult `figure_interpretation_guide.md` for interpretation
3. Consult `visualization_design.md` for detailed specifications
4. Check code comments in `create_publication_figures.py`

---

## Next Steps After Generation

1. **Review figures**: Check that they match expected patterns (see `figure_interpretation_guide.md`)
2. **Get feedback**: Share with collaborators
3. **Iterate**: Adjust colors, sizes, annotations as needed
4. **Finalize**: Export high-res versions for submission
5. **Archive**: Save final versions with paper submission

Good luck with your publication!
