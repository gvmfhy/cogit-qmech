#!/usr/bin/env python3
"""
Quantum Phase 1: Data Collection
Collect language model activations and encode as complex quantum states

Usage:
    python experiments/sentiment/quantum_phase1_collect.py --preset local
    python experiments/sentiment/quantum_phase1_collect.py --preset remote
    python experiments/sentiment/quantum_phase1_collect.py --preset tiny  # for testing
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import argparse

# Set deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np

# Import quantum components
from src.quantum_encoder import QuantumStateEncoder
from src.model_adapter_tl import ModelAdapterFactory
from config import QuantumConfig

# Deterministic seeding
torch.manual_seed(42)
np.random.seed(42)


class QuantumDataCollector:
    """Collect activations and encode as quantum states"""

    def __init__(self, config: QuantumConfig):
        self.config = config

        print("\n" + "=" * 70)
        print("QUANTUM PHASE 1: DATA COLLECTION")
        print("=" * 70)

        config.print_summary()

        # Load model
        print(f"\n[Loading {config.model_name} with TransformerLens]")
        # Use config device, but avoid MPS (TransformerLens has compatibility issues)
        import torch
        if config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif config.device == 'mps':
            print("⚠️  MPS has compatibility issues with TransformerLens, using CPU instead")
            device = 'cpu'
        else:
            device = config.device

        self.adapter = ModelAdapterFactory.create_adapter(config.model_name, device)
        print(f"✓ Model loaded on {device}")
        self.device = torch.device(device)

        # Validate config input_dim matches model
        if config.input_dim != self.adapter.hidden_dim:
            print(f"Warning: Config input_dim ({config.input_dim}) doesn't match model hidden_dim ({self.adapter.hidden_dim})")
            print(f"  Using model's hidden_dim: {self.adapter.hidden_dim}")
            config.input_dim = self.adapter.hidden_dim

        # Create quantum encoder
        print("\n[Creating Quantum State Encoder]")
        self.encoder = QuantumStateEncoder(
            input_dim=config.input_dim,
            quantum_dim=config.quantum_dim,
            seed=config.seed,
            device=self.device
        )

        # Load or generate prompts
        self.load_prompts()

    def load_prompts(self):
        """Load diverse prompts from file or generate new ones"""
        # Try prompts/ directory first (version controlled)
        prompt_file = ROOT / "prompts" / "diverse_prompts_50.json"

        # Fallback to data/ directory (legacy location)
        if not prompt_file.exists():
            prompt_file = ROOT / "data" / "sentiment_quantum" / "diverse_prompts_50.json"

        if prompt_file.exists():
            print(f"\n[Loading Prompts from {prompt_file.name}]")
            with open(prompt_file, 'r') as f:
                data = json.load(f)

            self.positive_prompts = data['positive_prompts'][:self.config.num_prompts]
            self.negative_prompts = data['negative_prompts'][:self.config.num_prompts]

            print(f"✓ Loaded {len(self.positive_prompts)} positive prompts")
            print(f"✓ Loaded {len(self.negative_prompts)} negative prompts")

        else:
            print("\n⚠️  Prompts file not found! Generating simple prompts...")
            self.generate_simple_prompts()

        # Show samples
        print("\nSample prompts:")
        print(f"  Positive: {self.positive_prompts[0]}")
        print(f"  Negative: {self.negative_prompts[0]}")

    def generate_simple_prompts(self):
        """Generate simple prompts if diverse ones aren't available"""
        self.positive_prompts = [
            f"I love {word}, it is so" for word in
            ["puppies", "sunshine", "chocolate", "music", "friends",
             "laughter", "success", "joy", "peace", "happiness"]
        ][:self.config.num_prompts]

        self.negative_prompts = [
            f"I hate {word}, it is so" for word in
            ["traffic", "delays", "errors", "problems", "failures",
             "sadness", "pain", "loss", "anger", "frustration"]
        ][:self.config.num_prompts]

    def collect_activations(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract language model activations from prompts"""

        print(f"\n[Extracting Activations from Layer {self.config.target_layer}]")

        positive_activations = []
        negative_activations = []

        # Collect positive activations
        print(f"\n🌟 Processing {len(self.positive_prompts)} positive prompts...")
        for i, prompt in enumerate(self.positive_prompts):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Progress: {i+1}/{len(self.positive_prompts)}")

            states = self.adapter.extract_hidden_states(prompt, [self.config.target_layer])
            activation = states[self.config.target_layer].cpu().numpy()
            positive_activations.append(activation)

        # Collect negative activations
        print(f"\n😞 Processing {len(self.negative_prompts)} negative prompts...")
        for i, prompt in enumerate(self.negative_prompts):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Progress: {i+1}/{len(self.negative_prompts)}")

            states = self.adapter.extract_hidden_states(prompt, [self.config.target_layer])
            activation = states[self.config.target_layer].cpu().numpy()
            negative_activations.append(activation)

        print(f"\n✅ Collected {len(positive_activations)} positive activations")
        print(f"✅ Collected {len(negative_activations)} negative activations")

        return positive_activations, negative_activations

    def encode_to_quantum_states(
        self,
        positive_activations: List[np.ndarray],
        negative_activations: List[np.ndarray]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode activations to quantum states"""

        print(f"\n[Encoding to Quantum States]")

        # Encode positive states
        positive_quantum = self.encoder.encode_batch(positive_activations)

        # Encode negative states
        negative_quantum = self.encoder.encode_batch(negative_activations)

        # Analyze separation
        stats = self.encoder.analyze_separation(positive_quantum, negative_quantum)

        return positive_quantum, negative_quantum, stats

    def save_quantum_data(
        self,
        positive_quantum: List[torch.Tensor],
        negative_quantum: List[torch.Tensor],
        stats: dict
    ) -> Path:
        """Save quantum states to file"""

        # Create output directory
        output_dir = ROOT / self.config.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for saving
        # Save complex tensors as {real, imag} pairs
        data = {
            'positive_quantum_states': [
                {
                    'real': state.real.tolist(),
                    'imag': state.imag.tolist()
                }
                for state in positive_quantum
            ],
            'negative_quantum_states': [
                {
                    'real': state.real.tolist(),
                    'imag': state.imag.tolist()
                }
                for state in negative_quantum
            ],
            'config': {
                'model_name': self.config.model_name,
                'model_identifier': self.config.model_identifier,
                'quantum_dim': self.config.quantum_dim,
                'input_dim': self.config.input_dim,
                'num_states': len(positive_quantum),
                'target_layer': self.config.target_layer,
                'seed': self.config.seed
            },
            'separation_stats': stats,
            'timestamp': timestamp
        }

        # Save quantum states (include model identifier in filename)
        model_id = self.config.model_identifier
        output_file = output_dir / f"quantum_states_{model_id}_{self.config.quantum_dim}d_{timestamp}.json"

        print(f"\n[Saving Quantum Data]")
        print(f"  File size estimate: {len(positive_quantum) * 2 * self.config.quantum_dim * 8 / 1024 / 1024:.1f} MB")

        with open(output_file, 'w') as f:
            json.dump(data, f)

        print(f"✓ Saved quantum states to {output_file.name}")

        # Save encoder projection matrix separately (for decoding)
        projection_file = output_dir / f"encoder_projection_{model_id}_{self.config.quantum_dim}d_{timestamp}.pt"
        self.encoder.save_projection_matrix(projection_file)

        # Create model-specific symlinks to 'latest' for easy access
        latest_states = output_dir / f"quantum_states_{model_id}_latest.json"
        latest_projection = output_dir / f"encoder_projection_{model_id}_latest.pt"

        # Remove old symlinks if they exist
        if latest_states.exists():
            latest_states.unlink()
        if latest_projection.exists():
            latest_projection.unlink()

        # Create new symlinks
        latest_states.symlink_to(output_file.name)
        latest_projection.symlink_to(projection_file.name)

        print(f"✓ Created symlink: quantum_states_{model_id}_latest.json")
        print(f"✓ Created symlink: encoder_projection_{model_id}_latest.pt")

        return output_file


def run_phase1(preset: str = 'local'):
    """Run Phase 1 data collection"""

    # Load configuration
    config = QuantumConfig.from_preset(preset)

    # Create collector
    collector = QuantumDataCollector(config)

    # Collect activations
    pos_acts, neg_acts = collector.collect_activations()

    # Encode to quantum states
    pos_quantum, neg_quantum, stats = collector.encode_to_quantum_states(pos_acts, neg_acts)

    # Save
    output_file = collector.save_quantum_data(pos_quantum, neg_quantum, stats)

    print("\n" + "=" * 70)
    print("🎉 QUANTUM PHASE 1 COMPLETE!")
    print("=" * 70)

    print(f"\nKey Achievements:")
    print(f"  ✓ Collected {len(pos_quantum)} quantum states per class")
    print(f"  ✓ Quantum dimension: {config.quantum_dim:,}-d complex")
    print(f"  ✓ States are normalized: ||ψ|| = 1")
    print(f"  ✓ Separation measured via quantum fidelity")

    print(f"\nNext Steps:")
    print(f"  → Phase 2: Train unitary operators on these quantum states")
    print(f"  → python experiments/sentiment/quantum_phase2_train.py --preset {preset}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Quantum Phase 1: Collect quantum states")
    parser.add_argument(
        '--preset',
        type=str,
        default='local',
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote'],
        help='Configuration preset (tiny/local/remote for GPT-2 124M, qwen_tiny/qwen_local for Qwen2.5-3B, qwen_remote for Qwen2.5-7B)'
    )

    args = parser.parse_args()

    print(f"\nRunning Phase 1 with preset: {args.preset.upper()}")
    run_phase1(preset=args.preset)


if __name__ == "__main__":
    main()
