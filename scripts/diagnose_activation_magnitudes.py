#!/usr/bin/env python3
"""
Quick diagnostic: Check if quantum steering produces reasonable activation magnitudes
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from config import QuantumConfig
from experiments.sentiment.quantum_phase3_test import QuantumInterventionSystem

def diagnose_magnitudes():
    """Check activation statistics before/after quantum intervention"""

    config = QuantumConfig.from_preset("qwen_local")
    system = QuantumInterventionSystem(config)

    test_prompt = "The weather today is"
    blend_ratio = 0.05
    max_tokens = 20

    print("=" * 70)
    print("ACTIVATION MAGNITUDE DIAGNOSTIC")
    print("=" * 70)
    print(f"\nPrompt: '{test_prompt}'")
    print(f"Blend ratio: {blend_ratio}")
    print(f"Target layer: {config.target_layer}")

    # Stats storage
    stats = {
        'original': [],
        'quantum_pos_to_neg': [],
        'quantum_neg_to_pos': []
    }

    # Modified hook to capture stats
    def capture_stats_hook(activation, hook, operator, blend, stat_key):
        # Original stats
        orig_mean = activation.mean().item()
        orig_std = activation.std().item()
        orig_min = activation.min().item()
        orig_max = activation.max().item()

        # Extract last token activation
        last_token_activation = activation[0, -1, :]

        # Encode to quantum
        quantum_state = system.encoder.encode_activation(last_token_activation)

        # Apply operator
        transformed_state = operator(quantum_state)

        # Blend
        blended_state = (1 - blend) * quantum_state + blend * transformed_state

        # Decode back
        decoded = system.decoder.decode_quantum_state(blended_state)

        # Replace last token
        modified_activation = activation.clone()
        modified_activation[0, -1, :] = decoded

        # Modified stats
        mod_mean = modified_activation.mean().item()
        mod_std = modified_activation.std().item()
        mod_min = modified_activation.min().item()
        mod_max = modified_activation.max().item()

        # Decoded activation stats (the actual intervention)
        dec_mean = decoded.mean().item()
        dec_std = decoded.std().item()
        dec_min = decoded.min().item()
        dec_max = decoded.max().item()

        stats[stat_key].append({
            'original': {
                'mean': orig_mean,
                'std': orig_std,
                'min': orig_min,
                'max': orig_max
            },
            'decoded': {
                'mean': dec_mean,
                'std': dec_std,
                'min': dec_min,
                'max': dec_max
            },
            'full_modified': {
                'mean': mod_mean,
                'std': mod_std,
                'min': mod_min,
                'max': mod_max
            }
        })

        return modified_activation

    # Test with pos_to_neg operator
    print("\n" + "-" * 70)
    print("Testing U_pos→neg operator...")
    print("-" * 70)

    def pos_to_neg_hook(activation, hook):
        return capture_stats_hook(
            activation, hook,
            system.operator_pos_to_neg,
            blend_ratio,
            'quantum_pos_to_neg'
        )

    hook_name = f"blocks.{config.target_layer}.hook_resid_post"
    tokens = system.adapter.model.to_tokens(test_prompt)

    with system.adapter.model.hooks(fwd_hooks=[(hook_name, pos_to_neg_hook)]):
        system.adapter.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            stop_at_eos=True,
            verbose=False
        )

    # Test with neg_to_pos operator
    print("\n" + "-" * 70)
    print("Testing U_neg→pos operator...")
    print("-" * 70)

    def neg_to_pos_hook(activation, hook):
        return capture_stats_hook(
            activation, hook,
            system.operator_neg_to_pos,
            blend_ratio,
            'quantum_neg_to_pos'
        )

    tokens = system.adapter.model.to_tokens(test_prompt)

    with system.adapter.model.hooks(fwd_hooks=[(hook_name, neg_to_pos_hook)]):
        system.adapter.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            stop_at_eos=True,
            verbose=False
        )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (averaged over generation steps)")
    print("=" * 70)

    for key in ['quantum_pos_to_neg', 'quantum_neg_to_pos']:
        if not stats[key]:
            continue

        print(f"\n{key.upper()}")
        print("-" * 70)

        # Average stats
        avg_orig_mean = sum(s['original']['mean'] for s in stats[key]) / len(stats[key])
        avg_orig_std = sum(s['original']['std'] for s in stats[key]) / len(stats[key])
        avg_orig_range = (
            sum(s['original']['max'] for s in stats[key]) / len(stats[key]) -
            sum(s['original']['min'] for s in stats[key]) / len(stats[key])
        )

        avg_dec_mean = sum(s['decoded']['mean'] for s in stats[key]) / len(stats[key])
        avg_dec_std = sum(s['decoded']['std'] for s in stats[key]) / len(stats[key])
        avg_dec_range = (
            sum(s['decoded']['max'] for s in stats[key]) / len(stats[key]) -
            sum(s['decoded']['min'] for s in stats[key]) / len(stats[key])
        )

        print(f"\nOriginal activation (full sequence):")
        print(f"  Mean: {avg_orig_mean:+.4f}")
        print(f"  Std:  {avg_orig_std:.4f}")
        print(f"  Range: {avg_orig_range:.4f}")

        print(f"\nDecoded activation (single token):")
        print(f"  Mean: {avg_dec_mean:+.4f}")
        print(f"  Std:  {avg_dec_std:.4f}")
        print(f"  Range: {avg_dec_range:.4f}")

        # Check for problems
        print(f"\nDIAGNOSTICS:")
        magnitude_ratio = abs(avg_dec_mean) / (abs(avg_orig_mean) + 1e-8)
        std_ratio = avg_dec_std / (avg_orig_std + 1e-8)

        if magnitude_ratio > 5 or magnitude_ratio < 0.2:
            print(f"  ⚠️  MAGNITUDE MISMATCH: Decoded mean is {magnitude_ratio:.1f}x original")
        else:
            print(f"  ✓ Magnitude reasonable ({magnitude_ratio:.2f}x original)")

        if std_ratio > 5 or std_ratio < 0.2:
            print(f"  ⚠️  STD MISMATCH: Decoded std is {std_ratio:.1f}x original")
        else:
            print(f"  ✓ Std reasonable ({std_ratio:.2f}x original)")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    diagnose_magnitudes()
