#!/usr/bin/env python3
"""
Quantum Phase 3: Test Interventions
Apply trained unitary operators to language model generation

Usage:
    python experiments/sentiment/quantum_phase3_test.py --preset local
    python experiments/sentiment/quantum_phase3_test.py --preset remote
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse

os.environ['PYTHONHASHSEED'] = '42'

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import torch
import numpy as np

from src.model_adapter_tl import TransformerLensAdapter
from src.quantum_encoder import QuantumStateEncoder
from src.quantum_decoder import QuantumStateDecoder
from src.unitary_operator import UnitaryOperator
from config import QuantumConfig, get_device

torch.manual_seed(42)
np.random.seed(42)


class QuantumInterventionSystem:
    """Test quantum interventions on language models"""

    def log_gpu_memory(self, stage=""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1e9
            total_gb = total / 1e9

            print(f"\n[GPU Memory {stage}]")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Free:      {free_gb:.2f} / {total_gb:.2f} GB")

    def __init__(self, config: QuantumConfig):
        self.config = config

        print("\n" + "=" * 70)
        print("QUANTUM PHASE 3: INTERVENTION TESTING")
        print("=" * 70)

        config.print_summary()

        # Load components
        self.load_model()
        self.log_gpu_memory("After model load")

        self.load_encoder()
        self.load_operators()
        self.log_gpu_memory("After operators load")

        self.create_decoder()

    def load_model(self):
        """Load language model from config"""
        print(f"\n[Loading {self.config.model_name}]")
        # Use config device with auto-detection
        import torch
        if self.config.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif self.config.device == 'mps':
            print("⚠️  MPS has compatibility issues with TransformerLens, using CPU instead")
            device = 'cpu'
        else:
            device = self.config.device

        self.adapter = TransformerLensAdapter(self.config.model_name, device)
        print(f"✓ {self.config.model_name} loaded on {device}")

    def load_encoder(self):
        """Load quantum encoder from Phase 1"""
        print("\n[Loading Quantum Encoder]")

        data_dir = ROOT / self.config.data_dir
        model_id = self.config.model_identifier
        projection_file = data_dir / f"encoder_projection_{model_id}_latest.pt"

        if not projection_file.exists():
            raise FileNotFoundError(
                f"Encoder projection for {model_id} not found! Run Phase 1 first."
            )

        self.encoder = QuantumStateEncoder.load_from_saved(projection_file)
        print("✓ Encoder loaded")

    def load_operator_smart(self, file_path: Path, quantum_dim: int, device='cuda'):
        """
        Load operator to GPU if memory available, else CPU

        This is device-aware and checks available GPU memory before loading.
        Critical for scaling from RTX 5090 (32GB) to H100 (80GB).
        """
        checkpoint = torch.load(file_path, map_location='cpu')
        operator = UnitaryOperator(quantum_dim=quantum_dim)
        operator.load_state_dict(checkpoint['model_state_dict'])

        if device == 'cuda' and torch.cuda.is_available():
            # Check available GPU memory
            gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info()
            operator_bytes = sum(p.numel() * p.element_size() for p in operator.parameters())

            # 1.2× safety margin for fragmentation
            if gpu_free_bytes > operator_bytes * 1.2:
                operator.to(device)
                print(f"  → Operator loaded to GPU ({operator_bytes/1e9:.2f} GB)")
                return operator, 'cuda'
            else:
                print(f"  ⚠️ Insufficient GPU memory ({gpu_free_bytes/1e9:.2f} GB free, need {operator_bytes/1e9:.2f} GB)")
                print(f"     Keeping operator on CPU (will be slower)")
                return operator, 'cpu'
        else:
            print(f"  → Operator on CPU (device={device})")
            return operator, 'cpu'

    def load_operators(self):
        """Load trained unitary operators from Phase 2"""
        print("\n[Loading Unitary Operators]")

        models_dir = ROOT / self.config.models_dir
        model_id = self.config.model_identifier

        # Load U_pos→neg
        pos_neg_file = models_dir / f"unitary_pos_to_neg_{model_id}_latest.pt"
        if not pos_neg_file.exists():
            raise FileNotFoundError(
                "Operator U_pos→neg not found! Run Phase 2 first."
            )

        # Device-aware loading
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Read checkpoint to get quantum_dim
        checkpoint_pos_neg = torch.load(pos_neg_file, map_location='cpu')
        quantum_dim = checkpoint_pos_neg['config']['quantum_dim']

        # Smart loading with memory check
        print(f"\nLoading U_pos→neg ({quantum_dim:,}-d):")
        self.operator_pos_to_neg, self.operator_pos_device = self.load_operator_smart(
            pos_neg_file, quantum_dim, device
        )
        self.operator_pos_to_neg.eval()
        print(f"✓ Loaded U_pos→neg on {self.operator_pos_device}")

        # Load U_neg→pos
        neg_pos_file = models_dir / f"unitary_neg_to_pos_{model_id}_latest.pt"
        if not neg_pos_file.exists():
            raise FileNotFoundError(
                "Operator U_neg→pos not found! Run Phase 2 first."
            )

        print(f"\nLoading U_neg→pos ({quantum_dim:,}-d):")
        self.operator_neg_to_pos, self.operator_neg_device = self.load_operator_smart(
            neg_pos_file, quantum_dim, device
        )
        self.operator_neg_to_pos.eval()
        print(f"✓ Loaded U_neg→pos on {self.operator_neg_device}")

        # Verify unitarity
        is_unitary_pos, dev_pos = self.operator_pos_to_neg.verify_unitarity()
        is_unitary_neg, dev_neg = self.operator_neg_to_pos.verify_unitarity()

        print(f"\n✓ U_pos→neg unitary: {is_unitary_pos} (deviation: {dev_pos:.6f})")
        print(f"✓ U_neg→pos unitary: {is_unitary_neg} (deviation: {dev_neg:.6f})")

    def create_decoder(self):
        """Create quantum decoder"""
        print("\n[Creating Quantum Decoder]")
        self.decoder = QuantumStateDecoder(self.encoder.projection)
        print("✓ Decoder created")

    def run_baseline(self, prompt: str, max_tokens: int = 25) -> str:
        """Run baseline generation without intervention"""
        tokens = self.adapter.model.to_tokens(prompt)
        output = self.adapter.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            stop_at_eos=True,
            verbose=False
        )

        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):].strip()

        return continuation

    def run_with_quantum_intervention(
        self,
        prompt: str,
        operator: UnitaryOperator,
        blend_ratio: float = 0.1,
        max_tokens: int = 25
    ) -> str:
        """
        Run generation with quantum intervention

        Args:
            prompt: Input prompt
            operator: Unitary operator to apply
            blend_ratio: Blending strength (0=none, 1=full)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text continuation
        """

        def quantum_intervention(activations, hook):
            """
            Apply quantum operator to activations (device-aware)

            This function is device-aware: it checks where the operator lives
            and moves quantum states to that device for computation.
            """

            # Determine operator device
            operator_device = next(operator.parameters()).device

            # 1. Encode activation to quantum state (always starts on CPU)
            quantum_state = self.encoder.encode_activation(activations.cpu().numpy())

            # 2. Move to operator's device if it's on GPU
            if operator_device.type == 'cuda':
                quantum_state = quantum_state.to(operator_device)

            # 3. Apply unitary operator (on whatever device it lives)
            with torch.no_grad():
                transformed_state = operator(quantum_state)

            # 4. Blend in quantum space (on same device as operator)
            if blend_ratio < 1.0:
                blended_state = self.decoder.quantum_blend(
                    quantum_state,
                    transformed_state,
                    blend_ratio=blend_ratio
                )
            else:
                blended_state = transformed_state

            # 5. Decode back to activation space (move to CPU for numpy conversion)
            decoded_activation = self.decoder.decode_quantum_state(
                blended_state.cpu(),
                method="real_component"
            )

            # 6. Reshape to match original activation shape
            original_shape = activations.shape
            if len(original_shape) == 3:
                batch_size, seq_len, dim = original_shape
                decoded_activation = decoded_activation.unsqueeze(0).unsqueeze(0)
                decoded_activation = decoded_activation.expand(batch_size, seq_len, dim)

            # 7. Final gentle blend in activation space (optional extra control)
            # This is insurance for coherence preservation
            gentle_blend = 0.5  # Extra damping
            final_activation = (
                (1 - gentle_blend) * activations.cpu() +
                gentle_blend * decoded_activation
            )

            return final_activation.to(activations.device)

        # Run generation with intervention
        hook_name = f"blocks.{self.config.target_layer}.hook_resid_post"

        tokens = self.adapter.model.to_tokens(prompt)

        with self.adapter.model.hooks(fwd_hooks=[(hook_name, quantum_intervention)]):
            output = self.adapter.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                stop_at_eos=True,
                verbose=False
            )

        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):].strip()

        return continuation

    def test_on_prompts(self, test_prompts: List[str]) -> List[Dict]:
        """Test interventions on multiple prompts"""

        results = []

        for prompt in test_prompts:
            print("\n" + "-" * 60)
            print(f"PROMPT: '{prompt}'")
            print("-" * 60)

            # Baseline
            print(f"\n📝 Baseline (no intervention):")
            baseline = self.run_baseline(prompt)
            print(f"   → {baseline}")

            # Test making it more negative
            print(f"\n😞 With U_pos→neg (make more negative):")
            interventions_negative = {}

            for ratio in self.config.blend_ratios:
                intervened = self.run_with_quantum_intervention(
                    prompt,
                    self.operator_pos_to_neg,
                    blend_ratio=ratio
                )
                interventions_negative[ratio] = intervened
                print(f"   Blend {ratio:.2f}: {intervened}")

            # Test making it more positive
            print(f"\n🌟 With U_neg→pos (make more positive):")
            interventions_positive = {}

            for ratio in self.config.blend_ratios:
                intervened = self.run_with_quantum_intervention(
                    prompt,
                    self.operator_neg_to_pos,
                    blend_ratio=ratio
                )
                interventions_positive[ratio] = intervened
                print(f"   Blend {ratio:.2f}: {intervened}")

            results.append({
                'prompt': prompt,
                'baseline': baseline,
                'interventions_negative': interventions_negative,
                'interventions_positive': interventions_positive
            })

        return results

    def save_results(self, results: List[Dict]):
        """Save intervention results"""

        results_dir = ROOT / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = results_dir / f"quantum_results_{self.config.quantum_dim}d_{timestamp}.json"

        data = {
            'results': results,
            'config': {
                'quantum_dim': self.config.quantum_dim,
                'blend_ratios': self.config.blend_ratios,
                'target_layer': self.config.target_layer
            },
            'timestamp': timestamp
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to {output_file.name}")

        # Create symlink to latest
        latest_file = results_dir / "quantum_results_latest.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)

        print(f"✓ Created symlink: quantum_results_latest.json")


def run_phase3(preset: str = 'local'):
    """Run Phase 3 intervention testing"""

    config = QuantumConfig.from_preset(preset)

    # Create intervention system
    system = QuantumInterventionSystem(config)

    # Define test prompts (neutral)
    test_prompts = [
        "The meeting this afternoon will",
        "I opened the envelope and found",
        "The restaurant downtown is",
        "My friend called to say",
        "The project manager announced that",
    ]

    print("\n" + "=" * 70)
    print("TESTING QUANTUM INTERVENTIONS")
    print("=" * 70)

    # Run tests
    results = system.test_on_prompts(test_prompts)

    # Save results
    system.save_results(results)

    print("\n" + "=" * 70)
    print("🎉 QUANTUM PHASE 3 COMPLETE!")
    print("=" * 70)

    print(f"\nKey Observations:")
    print(f"  → Compare baseline vs interventions")
    print(f"  → Check text coherence at different blend ratios")
    print(f"  → Observe sentiment shifts")

    print(f"\nNext Steps:")
    print(f"  → Phase 4: Test reversibility formally")
    print(f"  → python experiments/sentiment/test_reversibility.py --preset {preset}")
    print(f"  → Compare to classical HDC results")


def main():
    parser = argparse.ArgumentParser(description="Quantum Phase 3: Test interventions")
    parser.add_argument(
        '--preset',
        type=str,
        default='local',
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote'],
        help='Configuration preset'
    )

    args = parser.parse_args()

    print(f"\nRunning Phase 3 with preset: {args.preset.upper()}")
    run_phase3(preset=args.preset)


if __name__ == "__main__":
    main()
