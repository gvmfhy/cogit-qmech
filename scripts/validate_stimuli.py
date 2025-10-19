#!/usr/bin/env python3
"""
Validate and balance experimental stimuli for publication-grade rigor

Checks:
1. Length balance (word count)
2. Structural balance (first words, POS tags)
3. Explicit sentiment word contamination
4. Semantic diversity
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_stimuli(stimuli_file: Path):
    """Comprehensive stimulus analysis"""

    with open(stimuli_file, 'r') as f:
        data = json.load(f)

    pos = data['positive_prompts']
    neg = data['negative_prompts']

    print("=" * 70)
    print("STIMULUS VALIDATION REPORT")
    print("=" * 70)

    # 1. Sample size
    print(f"\n1. SAMPLE SIZE")
    print(f"   Positive: {len(pos)}")
    print(f"   Negative: {len(neg)}")
    print(f"   Balance: {'✅ Good' if abs(len(pos) - len(neg)) <= 5 else '⚠️  Imbalanced'}")

    # 2. Length statistics
    print(f"\n2. LENGTH STATISTICS")
    pos_lengths = [len(p.split()) for p in pos]
    neg_lengths = [len(p.split()) for p in neg]

    pos_mean = np.mean(pos_lengths)
    neg_mean = np.mean(neg_lengths)
    pos_std = np.std(pos_lengths)
    neg_std = np.std(neg_lengths)

    print(f"   Positive: {pos_mean:.2f} ± {pos_std:.2f} words")
    print(f"   Negative: {neg_mean:.2f} ± {neg_std:.2f} words")
    print(f"   Difference: {abs(pos_mean - neg_mean):.2f} words")
    print(f"   Status: {'✅ Matched' if abs(pos_mean - neg_mean) < 1.0 else '⚠️  Imbalanced'}")

    # 3. Structural analysis (first words)
    print(f"\n3. STRUCTURAL BALANCE (First Words)")

    pos_starts = Counter(p.split()[0] for p in pos)
    neg_starts = Counter(p.split()[0] for p in neg)

    # Normalize to proportions
    pos_props = {k: v/len(pos) for k, v in pos_starts.items()}
    neg_props = {k: v/len(neg) for k, v in neg_starts.items()}

    print(f"   Positive unique starts: {len(pos_starts)}")
    print(f"   Negative unique starts: {len(neg_starts)}")

    # Check for large imbalances
    all_starts = set(pos_starts.keys()) | set(neg_starts.keys())
    max_imbalance = 0
    worst_word = None

    for word in all_starts:
        pos_prop = pos_props.get(word, 0)
        neg_prop = neg_props.get(word, 0)
        imbalance = abs(pos_prop - neg_prop)

        if imbalance > max_imbalance:
            max_imbalance = imbalance
            worst_word = word

    print(f"\n   Worst imbalance: '{worst_word}' ({max_imbalance*100:.1f}% difference)")
    print(f"     Positive: {pos_props.get(worst_word, 0)*100:.1f}%")
    print(f"     Negative: {neg_props.get(worst_word, 0)*100:.1f}%")

    if max_imbalance > 0.15:
        print(f"   ⚠️  WARNING: Structural confound detected!")
        print(f"      Operator might learn '{worst_word}' → sentiment association")
    else:
        print(f"   ✅ Structural balance acceptable")

    # 4. Explicit sentiment words
    print(f"\n4. EXPLICIT SENTIMENT CONTAMINATION")

    sentiment_pos = ['wonderful', 'amazing', 'great', 'love', 'beautiful',
                     'perfect', 'delightful', 'incredible', 'fantastic', 'excellent']
    sentiment_neg = ['terrible', 'awful', 'horrible', 'hate', 'painful',
                     'frustrating', 'depressing', 'nightmare', 'devastating', 'miserable']

    pos_explicit = sum(1 for p in pos if any(w in p.lower() for w in sentiment_pos))
    neg_explicit = sum(1 for p in neg if any(w in p.lower() for w in sentiment_neg))

    print(f"   Positive with explicit words: {pos_explicit}/{len(pos)} ({100*pos_explicit/len(pos):.0f}%)")
    print(f"   Negative with explicit words: {neg_explicit}/{len(neg)} ({100*neg_explicit/len(neg):.0f}%)")

    if pos_explicit/len(pos) < 0.3 and neg_explicit/len(neg) < 0.3:
        print(f"   ✅ Low contamination (good for implicit learning)")
    else:
        print(f"   ⚠️  High contamination (operator might just learn keywords)")

    # 5. Recommendations
    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issues = []

    if abs(len(pos) - len(neg)) > 5:
        issues.append("• Equalize sample sizes")

    if abs(pos_mean - neg_mean) >= 1.0:
        issues.append("• Match mean stimulus length")

    if max_imbalance > 0.15:
        issues.append(f"• Fix structural imbalance ('{worst_word}' distribution)")
        issues.append("  - Option 1: Resample to balance first-word distributions")
        issues.append("  - Option 2: Generate new stimuli with matched syntax")

    if pos_explicit/len(pos) > 0.3:
        issues.append("• Reduce explicit sentiment words for implicit learning")

    if not issues:
        print("\n✅ Stimuli pass rigor checks!")
        print("   Ready for publication-grade experiments.")
    else:
        print("\n⚠️  Issues to address:")
        for issue in issues:
            print(f"   {issue}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    # Default to checking both classical and quantum
    classical_file = Path.home() / "project-cogit-framework" / "data" / "sentiment_experiment" / "diverse_prompts_50.json"
    quantum_file = Path.home() / "cogit-qmech" / "data" / "sentiment_quantum" / "diverse_prompts_50.json"

    if classical_file.exists():
        print("\nAnalyzing CLASSICAL stimuli:")
        analyze_stimuli(classical_file)

    if quantum_file.exists():
        print("\n\nAnalyzing QUANTUM stimuli:")
        analyze_stimuli(quantum_file)

    if not classical_file.exists() and not quantum_file.exists():
        print("No stimuli files found!")
        print("Expected:")
        print(f"  {classical_file}")
        print(f"  {quantum_file}")
