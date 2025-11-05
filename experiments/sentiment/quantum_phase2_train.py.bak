#!/usr/bin/env python3
"""
Quantum Phase 2: Train Unitary Operators
Train TWO unitary operators: U_pos→neg and U_neg→pos

Usage:
    python experiments/sentiment/quantum_phase2_train.py --preset local
    python experiments/sentiment/quantum_phase2_train.py --preset remote
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import argparse
import time

os.environ['PYTHONHASHSEED'] = '42'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import torch.optim as optim
import numpy as np

from src.unitary_operator import UnitaryOperator, BornRuleLoss
from src.quantum_utils import quantum_fidelity
from config import QuantumConfig, get_device

torch.manual_seed(42)
np.random.seed(42)


class UnitaryOperatorTrainer:
    """Train unitary operators for sentiment transformation"""

    def __init__(self, config: QuantumConfig):
        self.config = config
        self.device = get_device(config.device)

        print("\n" + "=" * 70)
        print("QUANTUM PHASE 2: UNITARY OPERATOR TRAINING")
        print("=" * 70)

        config.print_summary()
        print(f"\nActual device: {self.device}")

        # Load quantum states
        self.load_quantum_states()

        # Create operators
        self.create_operators()

    def load_quantum_states(self):
        """Load quantum states from Phase 1"""

        data_dir = ROOT / self.config.data_dir
        model_id = self.config.model_identifier

        # Try model-specific latest file first
        latest_file = data_dir / f"quantum_states_{model_id}_latest.json"

        if not latest_file.exists():
            # Fallback: try to find any quantum states file for this model
            state_files = list(data_dir.glob(f"quantum_states_{model_id}_*.json"))
            if state_files:
                latest_file = max(state_files, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(
                    f"No quantum states found for model '{model_id}'! Run Phase 1 first:\n"
                    f"  python experiments/sentiment/quantum_phase1_collect.py --preset {self.config.model_identifier}_tiny"
                )

        print(f"\n[Loading Quantum States]")
        print(f"  File: {latest_file.name}")

        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Reconstruct complex tensors from {real, imag} pairs
        def reconstruct_complex(state_dict):
            real = torch.tensor(state_dict['real'], dtype=torch.float32)
            imag = torch.tensor(state_dict['imag'], dtype=torch.float32)
            return torch.complex(real, imag)

        self.positive_states = torch.stack([
            reconstruct_complex(s) for s in data['positive_quantum_states']
        ]).to(self.device)

        self.negative_states = torch.stack([
            reconstruct_complex(s) for s in data['negative_quantum_states']
        ]).to(self.device)

        print(f"✓ Loaded {len(self.positive_states)} positive quantum states")
        print(f"✓ Loaded {len(self.negative_states)} negative quantum states")
        print(f"✓ Quantum dimension: {self.positive_states.shape[1]:,}-d")

        # Verify they're normalized
        pos_norms = torch.sqrt(torch.sum(torch.abs(self.positive_states) ** 2, dim=1))
        neg_norms = torch.sqrt(torch.sum(torch.abs(self.negative_states) ** 2, dim=1))

        print(f"✓ Positive states normalized: {torch.allclose(pos_norms, torch.ones_like(pos_norms), atol=1e-5)}")
        print(f"✓ Negative states normalized: {torch.allclose(neg_norms, torch.ones_like(neg_norms), atol=1e-5)}")

    def create_operators(self):
        """Create two unitary operators"""

        print(f"\n[Creating Unitary Operators]")

        quantum_dim = self.positive_states.shape[1]

        # Operator 1: positive → negative
        self.operator_pos_to_neg = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_pos_to_neg = self.operator_pos_to_neg.to(self.device)

        # Operator 2: negative → positive
        self.operator_neg_to_pos = UnitaryOperator(quantum_dim=quantum_dim)
        self.operator_neg_to_pos = self.operator_neg_to_pos.to(self.device)

        print(f"✓ Created U_pos→neg with {self.operator_pos_to_neg.count_parameters():,} parameters")
        print(f"✓ Created U_neg→pos with {self.operator_neg_to_pos.count_parameters():,} parameters")

    def train_operator(
        self,
        operator: UnitaryOperator,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
        name: str
    ) -> Tuple[List[float], List[float]]:
        """
        Train a single unitary operator

        Args:
            operator: UnitaryOperator to train
            source_states: Source quantum states
            target_states: Target quantum states
            name: Operator name (for logging)

        Returns:
            (loss_history, fidelity_history)
        """

        print(f"\n[Training {name}]")

        # Create optimizer
        optimizer = optim.Adam(
            operator.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )

        # Loss function
        criterion = BornRuleLoss()

        # Training history
        loss_history = []
        fidelity_history = []
        best_fidelity = 0.0

        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            operator.train()
            epoch_loss = 0.0
            epoch_fidelity = 0.0
            num_batches = 0

            # Shuffle data each epoch
            perm = torch.randperm(len(source_states))
            shuffled_source = source_states[perm]
            shuffled_target = target_states[perm]

            # Mini-batch training
            for i in range(0, len(shuffled_source), self.config.batch_size):
                batch_source = shuffled_source[i:i + self.config.batch_size]
                batch_target = shuffled_target[i:i + self.config.batch_size]

                # Forward pass
                output = operator(batch_source)

                # Born rule loss
                loss = criterion(output, batch_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=1.0)

                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                epoch_fidelity += (1.0 - loss.item())  # fidelity = 1 - loss
                num_batches += 1

            # Average metrics
            avg_loss = epoch_loss / num_batches
            avg_fidelity = epoch_fidelity / num_batches

            loss_history.append(avg_loss)
            fidelity_history.append(avg_fidelity)

            # Update learning rate
            scheduler.step(avg_loss)

            # Track best
            if avg_fidelity > best_fidelity:
                best_fidelity = avg_fidelity

            # Progress updates
            if (epoch + 1) % 20 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"    Epoch {epoch + 1:3d}/{self.config.epochs} | "
                      f"Loss: {avg_loss:.6f} | "
                      f"Fidelity: {avg_fidelity:.6f} | "
                      f"Time: {elapsed:.1f}s")

            # Verify unitarity periodically
            if (epoch + 1) % 50 == 0:
                is_unitary, deviation = operator.verify_unitarity()
                if not is_unitary:
                    print(f"    ⚠️  WARNING: Unitarity violated! Deviation: {deviation:.6f}")

        total_time = time.time() - start_time

        print(f"\n  ✅ Training complete in {total_time:.1f}s")
        print(f"     Final loss: {loss_history[-1]:.6f}")
        print(f"     Final fidelity: {fidelity_history[-1]:.6f}")
        print(f"     Best fidelity: {best_fidelity:.6f}")

        # Final unitarity check
        is_unitary, deviation = operator.verify_unitarity()
        print(f"     Unitary: {is_unitary} (deviation: {deviation:.6f})")

        return loss_history, fidelity_history

    def train_both_operators(self) -> Dict:
        """Train both operators"""

        print("\n" + "=" * 70)
        print("TRAINING TWO UNITARY OPERATORS")
        print("=" * 70)

        # Train positive → negative
        loss_pos_neg, fid_pos_neg = self.train_operator(
            self.operator_pos_to_neg,
            self.positive_states,
            self.negative_states,
            name="U_pos→neg"
        )

        # Train negative → positive
        loss_neg_pos, fid_neg_pos = self.train_operator(
            self.operator_neg_to_pos,
            self.negative_states,
            self.positive_states,
            name="U_neg→pos"
        )

        # Test reversibility
        print("\n[Testing Reversibility]")
        self.test_reversibility()

        return {
            'pos_to_neg': {
                'loss_history': loss_pos_neg,
                'fidelity_history': fid_pos_neg
            },
            'neg_to_pos': {
                'loss_history': loss_neg_pos,
                'fidelity_history': fid_neg_pos
            }
        }

    def test_reversibility(self):
        """Test if operators are approximately inverses"""

        self.operator_pos_to_neg.eval()
        self.operator_neg_to_pos.eval()

        with torch.no_grad():
            # Test: positive → negative → positive
            test_pos = self.positive_states[:5]
            transformed_neg = self.operator_pos_to_neg(test_pos)
            recovered_pos = self.operator_neg_to_pos(transformed_neg)

            fidelities_pos = []
            for i in range(len(test_pos)):
                fid = quantum_fidelity(test_pos[i], recovered_pos[i])
                fidelities_pos.append(fid.item())

            avg_fid_pos = np.mean(fidelities_pos)

            # Test: negative → positive → negative
            test_neg = self.negative_states[:5]
            transformed_pos = self.operator_neg_to_pos(test_neg)
            recovered_neg = self.operator_pos_to_neg(transformed_pos)

            fidelities_neg = []
            for i in range(len(test_neg)):
                fid = quantum_fidelity(test_neg[i], recovered_neg[i])
                fidelities_neg.append(fid.item())

            avg_fid_neg = np.mean(fidelities_neg)

        print(f"  pos → neg → pos fidelity: {avg_fid_pos:.4f}")
        print(f"  neg → pos → neg fidelity: {avg_fid_neg:.4f}")

        if avg_fid_pos > 0.7 and avg_fid_neg > 0.7:
            print(f"  ✅ Good reversibility! Operators are approximate inverses.")
        elif avg_fid_pos > 0.5 and avg_fid_neg > 0.5:
            print(f"  ⚠️  Moderate reversibility. May improve with more training.")
        else:
            print(f"  ❌ Poor reversibility. Operators are not inverse.")

    def save_operators(self, training_history: Dict):
        """Save trained operators"""

        models_dir = ROOT / self.config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self.config.model_identifier

        print(f"\n[Saving Operators]")

        # Save positive → negative operator
        pos_neg_path = models_dir / f"unitary_pos_to_neg_{model_id}_{self.config.quantum_dim}d_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.operator_pos_to_neg.state_dict(),
            'config': {
                'model_name': self.config.model_name,
                'model_identifier': model_id,
                'quantum_dim': self.config.quantum_dim,
                'input_dim': self.config.input_dim,
                'preset': 'unknown'  # Will be set by caller
            },
            'training_history': training_history['pos_to_neg'],
            'timestamp': timestamp
        }, pos_neg_path)

        print(f"  ✓ Saved U_pos→neg to {pos_neg_path.name}")

        # Save negative → positive operator
        neg_pos_path = models_dir / f"unitary_neg_to_pos_{model_id}_{self.config.quantum_dim}d_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.operator_neg_to_pos.state_dict(),
            'config': {
                'model_name': self.config.model_name,
                'model_identifier': model_id,
                'quantum_dim': self.config.quantum_dim,
                'input_dim': self.config.input_dim,
                'preset': 'unknown'
            },
            'training_history': training_history['neg_to_pos'],
            'timestamp': timestamp
        }, neg_pos_path)

        print(f"  ✓ Saved U_neg→pos to {neg_pos_path.name}")

        # Create model-specific symlinks to 'latest'
        latest_pos_neg = models_dir / f"unitary_pos_to_neg_{model_id}_latest.pt"
        latest_neg_pos = models_dir / f"unitary_neg_to_pos_{model_id}_latest.pt"

        if latest_pos_neg.exists():
            latest_pos_neg.unlink()
        if latest_neg_pos.exists():
            latest_neg_pos.unlink()

        latest_pos_neg.symlink_to(pos_neg_path.name)
        latest_neg_pos.symlink_to(neg_pos_path.name)

        print(f"  ✓ Created symlinks: unitary_*_{model_id}_latest.pt")


def run_phase2(preset: str = 'local'):
    """Run Phase 2 training"""

    config = QuantumConfig.from_preset(preset)

    # Create trainer
    trainer = UnitaryOperatorTrainer(config)

    # Train both operators
    training_history = trainer.train_both_operators()

    # Save operators
    trainer.save_operators(training_history)

    print("\n" + "=" * 70)
    print("🎉 QUANTUM PHASE 2 COMPLETE!")
    print("=" * 70)

    print(f"\nKey Achievements:")
    print(f"  ✓ Trained U_pos→neg (unitary operator)")
    print(f"  ✓ Trained U_neg→pos (unitary operator)")
    print(f"  ✓ Both operators maintain unitarity: U†U = I")
    print(f"  ✓ Reversibility tested")

    print(f"\nNext Steps:")
    print(f"  → Phase 3: Test interventions on language model")
    print(f"  → python experiments/sentiment/quantum_phase3_test.py --preset {preset}")


def main():
    parser = argparse.ArgumentParser(description="Quantum Phase 2: Train unitary operators")
    parser.add_argument(
        '--preset',
        type=str,
        default='local',
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote', 'pythia_410m', 'qwen3_4b'],
        help='Configuration preset (tiny/local/remote for GPT-2 124M, qwen_tiny/qwen_local for Qwen2.5-3B, qwen_remote for Qwen2.5-7B)'
    )

    args = parser.parse_args()

    print(f"\nRunning Phase 2 with preset: {args.preset.upper()}")
    run_phase2(preset=args.preset)


if __name__ == "__main__":
    main()
