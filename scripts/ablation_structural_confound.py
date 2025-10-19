#!/usr/bin/env python3
"""
Ablation Study: Test for Structural Confounds

Tests whether operator performance is driven by syntactic features
(e.g., "The" vs non-"The" prompts) or semantic content.

Usage:
    python scripts/ablation_structural_confound.py --preset local

This should be run AFTER Phase 3 (intervention testing)
"""

import os
import sys
from pathlib import Path
import argparse

os.environ['PYTHONHASHSEED'] = '42'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np
from collections import defaultdict

from src.model_adapter_tl import TransformerLensAdapter
from src.quantum_encoder import QuantumStateEncoder
from src.quantum_decoder import QuantumStateDecoder
from src.unitary_operator import UnitaryOperator
from src.quantum_utils import quantum_fidelity
from config import QuantumConfig

torch.manual_seed(42)
np.random.seed(42)


class StructuralConfoundAnalyzer:
    """Analyze whether structural features drive operator performance"""

    def __init__(self, config: QuantumConfig):
        self.config = config

        print("\n" + "=" * 70)
        print("ABLATION STUDY: STRUCTURAL CONFOUND ANALYSIS")
        print("=" * 70)
        print("\nResearch Question:")
        print("  Does the operator learn semantic sentiment, or does it")
        print("  exploit syntactic patterns (e.g., 'The' → negative)?")
        print("=" * 70)

        # Load quantum states
        self.load_quantum_states()

        # Load operators
        self.load_operators()

    def load_quantum_states(self):
        """Load quantum states from Phase 1"""

        data_dir = ROOT / self.config.data_dir
        latest_file = data_dir / "quantum_states_latest.json"

        if not latest_file.exists():
            raise FileNotFoundError("Quantum states not found! Run Phase 1 first.")

        print(f"\n[Loading Quantum States]")

        with open(latest_file, 'r') as f:
            data = json.load(f)

        def reconstruct_complex(state_dict):
            real = torch.tensor(state_dict['real'], dtype=torch.float32)
            imag = torch.tensor(state_dict['imag'], dtype=torch.float32)
            return torch.complex(real, imag)

        self.positive_states = [
            reconstruct_complex(s) for s in data['positive_quantum_states']
        ]

        self.negative_states = [
            reconstruct_complex(s) for s in data['negative_quantum_states']
        ]

        print(f"✓ Loaded {len(self.positive_states)} positive states")
        print(f"✓ Loaded {len(self.negative_states)} negative states")

        # Also load the original prompts to categorize
        self.load_original_prompts()

    def load_original_prompts(self):
        """Load original prompts to categorize by structure"""

        prompt_file = ROOT / "data" / "sentiment_quantum" / "diverse_prompts_50.json"

        if not prompt_file.exists():
            print("⚠️  Original prompts not found, cannot categorize by structure")
            self.positive_prompts = ["unknown"] * len(self.positive_states)
            self.negative_prompts = ["unknown"] * len(self.negative_states)
            return

        with open(prompt_file, 'r') as f:
            data = json.load(f)

        # Trim to match the number of collected states
        self.positive_prompts = data['positive_prompts'][:len(self.positive_states)]
        self.negative_prompts = data['negative_prompts'][:len(self.negative_states)]

        print(f"✓ Loaded original prompts for categorization")

    def load_operators(self):
        """Load trained operators"""

        models_dir = ROOT / self.config.models_dir

        print("\n[Loading Operators]")

        # Load U_pos→neg
        pos_neg_file = models_dir / "unitary_pos_to_neg_latest.pt"
        if not pos_neg_file.exists():
            raise FileNotFoundError("Operators not found! Run Phase 2 first.")

        checkpoint = torch.load(pos_neg_file, map_location='cpu')
        quantum_dim = checkpoint['config']['quantum_dim']

        self.operator_pos_to_neg = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_pos_to_neg.load_state_dict(checkpoint['model_state_dict'])
        self.operator_pos_to_neg.eval()

        # Load U_neg→pos
        neg_pos_file = models_dir / "unitary_neg_to_pos_latest.pt"
        checkpoint = torch.load(neg_pos_file, map_location='cpu')

        self.operator_neg_to_pos = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_neg_to_pos.load_state_dict(checkpoint['model_state_dict'])
        self.operator_neg_to_pos.eval()

        print(f"✓ Loaded operators")

    def categorize_by_structure(self):
        """Categorize states by syntactic structure"""

        print(f"\n[Categorizing by Structure]")

        # Positive states
        self.pos_the = []
        self.pos_nonthe = []
        self.pos_the_indices = []
        self.pos_nonthe_indices = []

        for i, (state, prompt) in enumerate(zip(self.positive_states, self.positive_prompts)):
            if prompt.startswith("The "):
                self.pos_the.append(state)
                self.pos_the_indices.append(i)
            else:
                self.pos_nonthe.append(state)
                self.pos_nonthe_indices.append(i)

        # Negative states
        self.neg_the = []
        self.neg_nonthe = []
        self.neg_the_indices = []
        self.neg_nonthe_indices = []

        for i, (state, prompt) in enumerate(zip(self.negative_states, self.negative_prompts)):
            if prompt.startswith("The "):
                self.neg_the.append(state)
                self.neg_the_indices.append(i)
            else:
                self.neg_nonthe.append(state)
                self.neg_nonthe_indices.append(i)

        print(f"\n  Positive:")
        print(f"    'The' prompts: {len(self.pos_the)}")
        print(f"    Non-'The' prompts: {len(self.pos_nonthe)}")

        print(f"\n  Negative:")
        print(f"    'The' prompts: {len(self.neg_the)}")
        print(f"    Non-'The' prompts: {len(self.neg_nonthe)}")

    def test_transformation_quality(self, source_states, target_states, operator, name):
        """
        Test how well operator transforms source → target

        Returns:
            Average fidelity to target class
        """

        if len(source_states) == 0:
            return 0.0

        fidelities = []

        with torch.no_grad():
            for source in source_states:
                # Transform
                transformed = operator(source)

                # Compute average fidelity to all target states
                target_fidelities = [
                    quantum_fidelity(transformed, target).item()
                    for target in target_states
                ]

                avg_fidelity = np.mean(target_fidelities)
                fidelities.append(avg_fidelity)

        mean_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)

        print(f"  {name}: {mean_fidelity:.4f} ± {std_fidelity:.4f}")

        return mean_fidelity

    def run_ablation_tests(self):
        """Run complete ablation analysis"""

        print("\n" + "=" * 70)
        print("ABLATION TESTS")
        print("=" * 70)

        results = {}

        # Test 1: Positive → Negative on 'The' vs non-'The'
        print("\n[Test 1: Positive → Negative Transformation Quality]")
        print("  If 'The' confound exists, 'The' prompts should transform better")

        # Stack states for batch processing
        pos_the_batch = torch.stack(self.pos_the) if self.pos_the else None
        pos_nonthe_batch = torch.stack(self.pos_nonthe) if self.pos_nonthe else None
        neg_all = self.negative_states

        if pos_the_batch is not None:
            fid_the = self.test_transformation_quality(
                self.pos_the, neg_all, self.operator_pos_to_neg,
                "Pos(The) → Neg"
            )
            results['pos_the_to_neg'] = fid_the
        else:
            results['pos_the_to_neg'] = 0.0

        if pos_nonthe_batch is not None:
            fid_nonthe = self.test_transformation_quality(
                self.pos_nonthe, neg_all, self.operator_pos_to_neg,
                "Pos(Non-The) → Neg"
            )
            results['pos_nonthe_to_neg'] = fid_nonthe
        else:
            results['pos_nonthe_to_neg'] = 0.0

        if pos_the_batch is not None and pos_nonthe_batch is not None:
            diff = abs(fid_the - fid_nonthe)
            print(f"\n  Difference: {diff:.4f}")

            if diff < 0.05:
                print(f"  ✅ No structural confound! (difference < 0.05)")
                print(f"     Operator learns semantic sentiment, not syntax.")
            elif diff < 0.1:
                print(f"  ⚠️  Slight structural effect (difference < 0.1)")
                print(f"     Mostly semantic, minor syntactic component.")
            else:
                print(f"  ❌ Structural confound detected! (difference > 0.1)")
                print(f"     Operator exploits 'The' pattern.")

            results['pos_to_neg_diff'] = diff

        # Test 2: Negative → Positive on 'The' vs non-'The'
        print("\n[Test 2: Negative → Positive Transformation Quality]")

        neg_the_batch = torch.stack(self.neg_the) if self.neg_the else None
        neg_nonthe_batch = torch.stack(self.neg_nonthe) if self.neg_nonthe else None
        pos_all = self.positive_states

        if neg_the_batch is not None:
            fid_the = self.test_transformation_quality(
                self.neg_the, pos_all, self.operator_neg_to_pos,
                "Neg(The) → Pos"
            )
            results['neg_the_to_pos'] = fid_the
        else:
            results['neg_the_to_pos'] = 0.0

        if neg_nonthe_batch is not None:
            fid_nonthe = self.test_transformation_quality(
                self.neg_nonthe, pos_all, self.operator_neg_to_pos,
                "Neg(Non-The) → Pos"
            )
            results['neg_nonthe_to_pos'] = fid_nonthe
        else:
            results['neg_nonthe_to_pos'] = 0.0

        if neg_the_batch is not None and neg_nonthe_batch is not None:
            diff = abs(fid_the - fid_nonthe)
            print(f"\n  Difference: {diff:.4f}")

            if diff < 0.05:
                print(f"  ✅ No structural confound!")
            elif diff < 0.1:
                print(f"  ⚠️  Slight structural effect")
            else:
                print(f"  ❌ Structural confound detected!")

            results['neg_to_pos_diff'] = diff

        # Overall assessment
        print("\n" + "=" * 70)
        print("OVERALL ASSESSMENT")
        print("=" * 70)

        avg_diff = np.mean([
            results.get('pos_to_neg_diff', 0),
            results.get('neg_to_pos_diff', 0)
        ])

        print(f"\nAverage structural effect: {avg_diff:.4f}")

        if avg_diff < 0.05:
            print("\n✅ CONCLUSION: Operators learn SEMANTIC sentiment")
            print("   The 'The' imbalance does NOT drive operator performance.")
            print("   Safe to proceed with current stimuli for publication.")
        elif avg_diff < 0.1:
            print("\n⚠️  CONCLUSION: Mostly semantic, minor syntactic component")
            print("   Operators primarily learn sentiment, with slight syntactic bias.")
            print("   Acceptable for publication with disclosure in methods.")
        else:
            print("\n❌ CONCLUSION: Operators exploit SYNTACTIC patterns")
            print("   The 'The' imbalance DOES drive operator performance.")
            print("   Consider generating balanced stimuli for stronger claims.")

        return results

    def save_results(self, results):
        """Save ablation results"""

        results_dir = ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = results_dir / f"ablation_structural_{timestamp}.json"

        data = {
            'results': results,
            'stimulus_counts': {
                'positive_the': len(self.pos_the),
                'positive_nonthe': len(self.pos_nonthe),
                'negative_the': len(self.neg_the),
                'negative_nonthe': len(self.neg_nonthe)
            },
            'config': {
                'quantum_dim': self.config.quantum_dim
            },
            'timestamp': timestamp
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to {output_file.name}")


def run_ablation_study(preset: str = 'local'):
    """Run structural confound ablation study"""

    config = QuantumConfig.from_preset(preset)

    analyzer = StructuralConfoundAnalyzer(config)

    # Categorize by structure
    analyzer.categorize_by_structure()

    # Run tests
    results = analyzer.run_ablation_tests()

    # Save results
    analyzer.save_results(results)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    print("\nThis analysis should be included in your methods section:")
    print("  'To test for structural confounds, we performed ablation")
    print("   analysis comparing operator performance on prompts beginning")
    print("   with 'The' versus other structures. Results showed...'")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: Test for structural confounds"
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='local',
        choices=['tiny', 'local', 'remote'],
        help='Configuration preset (should match Phase 1/2)'
    )

    args = parser.parse_args()

    print(f"\nRunning ablation study with preset: {args.preset.upper()}")
    print("Note: This should be run AFTER Phase 2 (training)")

    run_ablation_study(preset=args.preset)


if __name__ == "__main__":
    main()
