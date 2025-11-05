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
from typing import List, Tuple, Optional
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

    def __init__(self, config: QuantumConfig, prompts_path: Optional[Path] = None, num_prompts_override: Optional[int] = None, quantum_ratio: Optional[float] = None):
        self.config = config
        self._prompts_path = prompts_path
        self._num_prompts_override = num_prompts_override
        self._quantum_ratio_override = quantum_ratio

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

        # Optional: override quantum_dim from ratio once input_dim is known
        if self._quantum_ratio_override is not None and self._quantum_ratio_override > 0:
            new_qdim = int(round(config.input_dim * self._quantum_ratio_override))
            if new_qdim != config.quantum_dim:
                print(f"  → Overriding quantum_dim via ratio {self._quantum_ratio_override} → {new_qdim}")
                config.quantum_dim = new_qdim

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
        # Prefer explicitly provided prompts path
        if self._prompts_path is not None:
            prompt_file = self._prompts_path
        else:
            # Try prompts/ directory first (version controlled)
            prompt_file = ROOT / "prompts" / "diverse_prompts_50.json"

        # Fallback to data/ directory (legacy location)
        if not prompt_file.exists():
            prompt_file = ROOT / "data" / "sentiment_quantum" / "diverse_prompts_50.json"

        if prompt_file.exists():
            print(f"\n[Loading Prompts from {prompt_file.name}]")
            with open(prompt_file, 'r') as f:
                data = json.load(f)

            n = self._num_prompts_override or self.config.num_prompts
            self.positive_prompts = data.get('positive_prompts', [])[:n]
            self.negative_prompts = data.get('negative_prompts', [])[:n]

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
        n = self._num_prompts_override or self.config.num_prompts
        self.positive_prompts = [
            f"I love {word}, it is so" for word in
            ["puppies", "sunshine", "chocolate", "music", "friends",
             "laughter", "success", "joy", "peace", "happiness"]
        ][:n]

        self.negative_prompts = [
            f"I hate {word}, it is so" for word in
            ["traffic", "delays", "errors", "problems", "failures",
             "sadness", "pain", "loss", "anger", "frustration"]
        ][:n]

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

    def save_quantum_data_with_layer(
        self,
        positive_quantum: List[torch.Tensor],
        negative_quantum: List[torch.Tensor],
        stats: dict,
        layer_idx: int
    ) -> Path:
        """Save quantum states with layer identifier in filename"""

        # Create output directory
        output_dir = ROOT / self.config.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for saving
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
                'target_layer': layer_idx,  # Use the specific layer
                'seed': self.config.seed
            },
            'separation_stats': stats,
            'timestamp': timestamp
        }

        # Include layer in filename
        model_id = self.config.model_identifier
        output_file = output_dir / f"quantum_states_{model_id}_layer{layer_idx}_{self.config.quantum_dim}d_{timestamp}.json"

        print(f"\n[Saving Quantum Data for Layer {layer_idx}]")

        with open(output_file, 'w') as f:
            json.dump(data, f)

        print(f"✓ Saved to {output_file.name}")

        # Save encoder projection for this layer
        projection_file = output_dir / f"encoder_projection_{model_id}_layer{layer_idx}_{self.config.quantum_dim}d_{timestamp}.pt"
        self.encoder.save_projection_matrix(projection_file)

        return output_file


def run_phase1(preset: str = 'local', *, prompts_path: Optional[str] = None, num_prompts: Optional[int] = None):
    """Run Phase 1 data collection"""

    # Load configuration
    config = QuantumConfig.from_preset(preset)

    # Check if this is a layer sweep
    if hasattr(config, 'test_layers') and config.test_layers is not None:
        return run_layer_sweep(config, preset, prompts_path=prompts_path, num_prompts=num_prompts)
    else:
        return run_single_layer(config, preset, prompts_path=prompts_path, num_prompts=num_prompts)


def run_layer_sweep(config: QuantumConfig, preset: str, *, prompts_path: Optional[str] = None, num_prompts: Optional[int] = None):
    """Run Phase 1 for multiple layers to find optimal separation"""

    print("\n" + "=" * 70)
    print("LAYER SWEEP MODE")
    print("=" * 70)
    print(f"\nTesting layers: {config.test_layers}")
    print(f"This will help identify which layer has best pos/neg separation\n")

    layer_results = []

    for layer_idx in config.test_layers:
        print("\n" + "=" * 70)
        print(f"TESTING LAYER {layer_idx}")
        print("=" * 70)

        # Create a temporary config with this specific layer
        layer_config = QuantumConfig.from_preset(preset)
        layer_config.target_layer = layer_idx

        # Create collector for this layer
        collector = QuantumDataCollector(
            layer_config,
            prompts_path=Path(prompts_path) if prompts_path else None,
            num_prompts_override=num_prompts
        )

        # Collect activations at this layer
        pos_acts, neg_acts = collector.collect_activations()

        # Encode to quantum states
        pos_quantum, neg_quantum, stats = collector.encode_to_quantum_states(pos_acts, neg_acts)

        # Save with layer identifier
        output_file = collector.save_quantum_data_with_layer(
            pos_quantum, neg_quantum, stats, layer_idx
        )

        # Store results
        layer_results.append({
            'layer': layer_idx,
            'stats': stats,
            'output_file': output_file
        })

        print(f"\n✅ Layer {layer_idx} complete")
        print(f"   Centroid fidelity: {stats['centroid_fidelity']:.6f}")
        print(f"   Separation gap: {stats['separation_gap']:.6f}")

    # Print summary comparison
    print("\n" + "=" * 70)
    print("LAYER SWEEP SUMMARY")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'Centroid Fidelity':<20} {'Separation Gap':<20} {'Quality'}")
    print("-" * 70)

    for result in layer_results:
        layer = result['layer']
        centroid_fid = result['stats']['centroid_fidelity']
        sep_gap = result['stats']['separation_gap']

        # Quality assessment
        if sep_gap > 0.05:
            quality = "✅ EXCELLENT"
        elif sep_gap > 0.02:
            quality = "✅ GOOD"
        elif sep_gap > 0.01:
            quality = "⚠️  MODERATE"
        else:
            quality = "❌ POOR"

        print(f"{layer:<8} {centroid_fid:<20.6f} {sep_gap:<20.6f} {quality}")

    # Find best layer
    best_result = min(layer_results, key=lambda r: r['stats']['centroid_fidelity'])
    best_layer = best_result['layer']
    best_gap = best_result['stats']['separation_gap']

    print("\n" + "=" * 70)
    print(f"🎯 BEST LAYER: {best_layer}")
    print(f"   Separation gap: {best_gap:.6f}")
    print("=" * 70)

    print(f"\nNext Steps:")
    print(f"  1. Use layer {best_layer} for training (update config.target_layer)")
    print(f"  2. Run Phase 2: python experiments/sentiment/quantum_phase2_train.py --preset {preset}")
    print(f"  3. If all layers show poor separation (<1%), consider:")
    print(f"     - Testing a classical additive steering baseline")
    print(f"     - Using a different model or prompts")

    return best_result['output_file']


def run_single_layer(config: QuantumConfig, preset: str, *, prompts_path: Optional[str] = None, num_prompts: Optional[int] = None):
    """Run Phase 1 for a single layer (original behavior)"""

    # Create collector
    collector = QuantumDataCollector(
        config,
        prompts_path=Path(prompts_path) if prompts_path else None,
        num_prompts_override=num_prompts
    )

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
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote', 'pythia_410m', 'pythia_test_layers', 'qwen3_4b', 'qwen3_4b_test_layers'],
        help='Configuration preset (tiny/local/remote for GPT-2 124M, qwen_tiny/qwen_local for Qwen2.5-3B, qwen_remote for Qwen2.5-7B)'
    )
    parser.add_argument('--prompts', type=str, default='', help='Path to prompts JSON (expects positive_prompts and negative_prompts)')
    parser.add_argument('--num-prompts', type=int, default=0, help='Override number of prompts to load per class')
    parser.add_argument('--model', type=str, default='', help='Override model name (e.g., EleutherAI/pythia-410m, Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--quantum-dim', type=int, default=0, help='Override quantum dimension directly')
    parser.add_argument('--quantum-ratio', type=float, default=0.0, help='Set quantum_dim = round(input_dim * quantum_ratio)')

    args = parser.parse_args()

    print(f"\nRunning Phase 1 with preset: {args.preset.upper()}")
    # Allow model and quantum dim overrides via CLI
    cfg = QuantumConfig.from_preset(args.preset)
    if args.model:
        cfg.model_name = args.model
    if args.quantum_dim and args.quantum_dim > 0:
        cfg.quantum_dim = args.quantum_dim

    # Execute with overrides
    quantum_ratio = args.quantum_ratio if args.quantum_ratio and args.quantum_ratio > 0 else None
    # We need to pass a config instance, so call run_single_layer/run_layer_sweep directly
    if hasattr(cfg, 'test_layers') and cfg.test_layers is not None:
        run_layer_sweep(
            cfg,
            preset=args.preset,
            prompts_path=args.prompts if args.prompts else None,
            num_prompts=args.num_prompts if args.num_prompts and args.num_prompts > 0 else None,
        )
    else:
        collector = QuantumDataCollector(
            cfg,
            prompts_path=Path(args.prompts) if args.prompts else None,
            num_prompts_override=args.num_prompts if args.num_prompts and args.num_prompts > 0 else None,
            quantum_ratio=quantum_ratio,
        )
        pos_acts, neg_acts = collector.collect_activations()
        pos_q, neg_q, stats = collector.encode_to_quantum_states(pos_acts, neg_acts)
        collector.save_quantum_data(pos_q, neg_q, stats)


if __name__ == "__main__":
    main()
