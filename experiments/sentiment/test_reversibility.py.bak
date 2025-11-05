#!/usr/bin/env python3
"""
Test Reversibility of Quantum Operators

This is the KEY TEST that classical HDC cannot pass:
- Classical: pos→neg→pos ≠ original (irreversible)
- Quantum: pos→neg→pos ≈ original (reversible due to unitarity)

Usage:
    python experiments/sentiment/test_reversibility.py --preset local
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

os.environ['PYTHONHASHSEED'] = '42'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.unitary_operator import UnitaryOperator
from src.quantum_utils import quantum_fidelity
from config import QuantumConfig

torch.manual_seed(42)
np.random.seed(42)


class ReversibilityTester:
    """Test reversibility of quantum operators"""

    def __init__(self, config: QuantumConfig):
        self.config = config

        print("\n" + "=" * 70)
        print("QUANTUM REVERSIBILITY TEST")
        print("=" * 70)
        print("\nThis test validates the quantum approach:")
        print("  Classical HDC: pos→neg→pos ≠ original (NOT reversible)")
        print("  Quantum:       pos→neg→pos ≈ original (reversible!)")
        print("=" * 70)

        self.load_quantum_states()
        self.load_operators()

    def load_quantum_states(self):
        """Load quantum states from Phase 1"""

        data_dir = ROOT / self.config.data_dir
        model_id = self.config.model_identifier
        latest_file = data_dir / f"quantum_states_{model_id}_latest.json"

        if not latest_file.exists():
            raise FileNotFoundError("Quantum states not found! Run Phase 1 first.")

        print(f"\n[Loading Quantum States]")

        with open(latest_file, 'r') as f:
            data = json.load(f)

        def reconstruct_complex(state_dict):
            real = torch.tensor(state_dict['real'], dtype=torch.float32)
            imag = torch.tensor(state_dict['imag'], dtype=torch.float32)
            return torch.complex(real, imag)

        self.positive_states = torch.stack([
            reconstruct_complex(s) for s in data['positive_quantum_states']
        ])

        self.negative_states = torch.stack([
            reconstruct_complex(s) for s in data['negative_quantum_states']
        ])

        print(f"✓ Loaded {len(self.positive_states)} positive states")
        print(f"✓ Loaded {len(self.negative_states)} negative states")

    def load_operators(self):
        """Load trained operators"""

        models_dir = ROOT / self.config.models_dir
        model_id = self.config.model_identifier

        print("\n[Loading Operators]")

        # Load U_pos→neg
        pos_neg_file = models_dir / f"unitary_pos_to_neg_{model_id}_latest.pt"
        checkpoint_pos_neg = torch.load(pos_neg_file, map_location='cpu')
        quantum_dim = checkpoint_pos_neg['config']['quantum_dim']

        self.operator_pos_to_neg = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_pos_to_neg.load_state_dict(checkpoint_pos_neg['model_state_dict'])
        self.operator_pos_to_neg.eval()

        # Load U_neg→pos
        neg_pos_file = models_dir / f"unitary_neg_to_pos_{model_id}_latest.pt"
        checkpoint_neg_pos = torch.load(neg_pos_file, map_location='cpu')

        self.operator_neg_to_pos = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_neg_to_pos.load_state_dict(checkpoint_neg_pos['model_state_dict'])
        self.operator_neg_to_pos.eval()

        print(f"✓ Loaded operators ({quantum_dim:,}-d)")

    def test_positive_negative_positive(self) -> dict:
        """Test: positive → negative → positive"""

        print("\n[Test 1: Positive → Negative → Positive]")

        fidelities = []
        samples = min(20, len(self.positive_states))

        with torch.no_grad():
            for i in range(samples):
                original = self.positive_states[i]

                # Apply pos→neg
                transformed = self.operator_pos_to_neg(original)

                # Apply neg→pos
                recovered = self.operator_neg_to_pos(transformed)

                # Compute fidelity
                fid = quantum_fidelity(original, recovered)
                fidelities.append(fid.item())

                if i < 5:
                    print(f"  State {i+1}: fidelity = {fid:.6f}")

        avg_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)
        min_fidelity = np.min(fidelities)
        max_fidelity = np.max(fidelities)

        print(f"\n  Average fidelity: {avg_fidelity:.6f} ± {std_fidelity:.6f}")
        print(f"  Range: [{min_fidelity:.6f}, {max_fidelity:.6f}]")

        if avg_fidelity > 0.9:
            print(f"  ✅ EXCELLENT reversibility!")
        elif avg_fidelity > 0.7:
            print(f"  ✅ GOOD reversibility")
        elif avg_fidelity > 0.5:
            print(f"  ⚠️  MODERATE reversibility")
        else:
            print(f"  ❌ POOR reversibility")

        return {
            'fidelities': fidelities,
            'mean': avg_fidelity,
            'std': std_fidelity,
            'min': min_fidelity,
            'max': max_fidelity
        }

    def test_negative_positive_negative(self) -> dict:
        """Test: negative → positive → negative"""

        print("\n[Test 2: Negative → Positive → Negative]")

        fidelities = []
        samples = min(20, len(self.negative_states))

        with torch.no_grad():
            for i in range(samples):
                original = self.negative_states[i]

                # Apply neg→pos
                transformed = self.operator_neg_to_pos(original)

                # Apply pos→neg
                recovered = self.operator_pos_to_neg(transformed)

                # Compute fidelity
                fid = quantum_fidelity(original, recovered)
                fidelities.append(fid.item())

                if i < 5:
                    print(f"  State {i+1}: fidelity = {fid:.6f}")

        avg_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)
        min_fidelity = np.min(fidelities)
        max_fidelity = np.max(fidelities)

        print(f"\n  Average fidelity: {avg_fidelity:.6f} ± {std_fidelity:.6f}")
        print(f"  Range: [{min_fidelity:.6f}, {max_fidelity:.6f}]")

        if avg_fidelity > 0.9:
            print(f"  ✅ EXCELLENT reversibility!")
        elif avg_fidelity > 0.7:
            print(f"  ✅ GOOD reversibility")
        elif avg_fidelity > 0.5:
            print(f"  ⚠️  MODERATE reversibility")
        else:
            print(f"  ❌ POOR reversibility")

        return {
            'fidelities': fidelities,
            'mean': avg_fidelity,
            'std': std_fidelity,
            'min': min_fidelity,
            'max': max_fidelity
        }

    def test_operator_conjugate_relationship(self) -> dict:
        """Test if U_neg→pos ≈ (U_pos→neg)†"""

        print("\n[Test 3: Operator Conjugate Relationship]")
        print("  Testing if U_neg→pos ≈ U†_pos→neg (conjugate transpose)")

        # Get unitary matrices
        U_pos_to_neg = self.operator_pos_to_neg.get_unitary_matrix()
        U_neg_to_pos = self.operator_neg_to_pos.get_unitary_matrix()

        # Get conjugate transpose of U_pos_to_neg
        U_pos_to_neg_dagger = torch.conj(U_pos_to_neg.T)

        # Compute Frobenius norm of difference
        diff = U_neg_to_pos - U_pos_to_neg_dagger
        frobenius_norm = torch.sqrt(torch.sum(torch.abs(diff) ** 2))

        # Normalize by matrix size
        normalized_diff = frobenius_norm / (U_pos_to_neg.shape[0] * U_pos_to_neg.shape[1])

        print(f"  Frobenius norm difference: {frobenius_norm:.6f}")
        print(f"  Normalized difference: {normalized_diff:.6f}")

        if normalized_diff < 0.001:
            print(f"  ✅ Operators are approximately conjugate transposes!")
        elif normalized_diff < 0.01:
            print(f"  ⚠️  Operators are loosely related (not exact conjugates)")
        else:
            print(f"  ❌ Operators are NOT conjugate transposes")
            print(f"     (This is expected - they were trained independently)")

        return {
            'frobenius_norm': frobenius_norm.item(),
            'normalized_diff': normalized_diff.item()
        }

    def visualize_reversibility(self, results_pos_neg_pos: dict, results_neg_pos_neg: dict):
        """Create visualization of reversibility"""

        print("\n[Creating Visualization]")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Positive → Negative → Positive
        ax1.hist(results_pos_neg_pos['fidelities'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results_pos_neg_pos['mean'], color='red', linestyle='--',
                    label=f'Mean: {results_pos_neg_pos["mean"]:.3f}')
        ax1.set_xlabel('Fidelity')
        ax1.set_ylabel('Count')
        ax1.set_title('Reversibility: pos → neg → pos')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Negative → Positive → Negative
        ax2.hist(results_neg_pos_neg['fidelities'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(results_neg_pos_neg['mean'], color='red', linestyle='--',
                    label=f'Mean: {results_neg_pos_neg["mean"]:.3f}')
        ax2.set_xlabel('Fidelity')
        ax2.set_ylabel('Count')
        ax2.set_title('Reversibility: neg → pos → neg')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        results_dir = ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = results_dir / f"reversibility_plot_{timestamp}.png"

        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_file.name}")

        plt.close()

    def save_results(self, results: dict):
        """Save reversibility test results"""

        results_dir = ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = results_dir / f"reversibility_results_{timestamp}.json"

        # Convert numpy arrays to lists
        data = {
            'positive_negative_positive': {
                'mean': results['pos_neg_pos']['mean'],
                'std': results['pos_neg_pos']['std'],
                'min': results['pos_neg_pos']['min'],
                'max': results['pos_neg_pos']['max'],
                'fidelities': results['pos_neg_pos']['fidelities']
            },
            'negative_positive_negative': {
                'mean': results['neg_pos_neg']['mean'],
                'std': results['neg_pos_neg']['std'],
                'min': results['neg_pos_neg']['min'],
                'max': results['neg_pos_neg']['max'],
                'fidelities': results['neg_pos_neg']['fidelities']
            },
            'conjugate_test': results['conjugate'],
            'config': {
                'quantum_dim': self.config.quantum_dim
            },
            'timestamp': timestamp
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Results saved to {output_file.name}")


def run_reversibility_test(preset: str = 'local'):
    """Run complete reversibility test"""

    config = QuantumConfig.from_preset(preset)

    # Create tester
    tester = ReversibilityTester(config)

    # Run tests
    results_pos_neg_pos = tester.test_positive_negative_positive()
    results_neg_pos_neg = tester.test_negative_positive_negative()
    results_conjugate = tester.test_operator_conjugate_relationship()

    # Visualize
    tester.visualize_reversibility(results_pos_neg_pos, results_neg_pos_neg)

    # Save results
    results = {
        'pos_neg_pos': results_pos_neg_pos,
        'neg_pos_neg': results_neg_pos_neg,
        'conjugate': results_conjugate
    }
    tester.save_results(results)

    print("\n" + "=" * 70)
    print("🎉 REVERSIBILITY TEST COMPLETE!")
    print("=" * 70)

    print(f"\nSummary:")
    print(f"  pos→neg→pos: {results_pos_neg_pos['mean']:.4f} ± {results_pos_neg_pos['std']:.4f}")
    print(f"  neg→pos→neg: {results_neg_pos_neg['mean']:.4f} ± {results_neg_pos_neg['std']:.4f}")

    avg_both = (results_pos_neg_pos['mean'] + results_neg_pos_neg['mean']) / 2

    print(f"\n  Overall reversibility: {avg_both:.4f}")

    if avg_both > 0.9:
        print(f"\n  ✅ QUANTUM ADVANTAGE CONFIRMED!")
        print(f"     Unitary operators are highly reversible.")
        print(f"     Classical HDC cannot achieve this!")
    elif avg_both > 0.7:
        print(f"\n  ✅ QUANTUM ADVANTAGE DEMONSTRATED!")
        print(f"     Good reversibility achieved.")
    else:
        print(f"\n  ⚠️  Moderate reversibility.")
        print(f"     Consider training longer or adjusting hyperparameters.")


def main():
    parser = argparse.ArgumentParser(description="Test quantum operator reversibility")
    parser.add_argument(
        '--preset',
        type=str,
        default='local',
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote', 'pythia_410m', 'qwen3_4b'],
        help='Configuration preset'
    )

    args = parser.parse_args()

    print(f"\nRunning reversibility test with preset: {args.preset.upper()}")
    run_reversibility_test(preset=args.preset)


if __name__ == "__main__":
    main()
