#!/usr/bin/env python3
"""
Configuration for Cogit-QMech
Handles local (M1 MacBook) vs remote (Cloud GPU) settings
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class QuantumConfig:
    """
    Configuration for quantum cognitive operators

    Presets:
    - 'local': Optimized for M1 MacBook 16GB (quantum_dim=2000)
    - 'remote': Full-scale for cloud GPU (quantum_dim=10000)
    - 'tiny': Fast testing/debugging (quantum_dim=500)
    """

    # Model dimensions
    input_dim: int = 768          # GPT-2 activation dimension
    quantum_dim: int = 2000       # Quantum state dimension

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 10
    epochs: int = 100

    # Data collection
    num_prompts: int = 50         # Per sentiment class
    target_layer: int = 6         # GPT-2 layer to extract

    # Intervention
    blend_ratios: list = None     # Will default to [0.02, 0.05, 0.1, 0.2]

    # Device
    device: str = 'auto'          # 'auto', 'cpu', 'mps', 'cuda'

    # Paths
    data_dir: str = 'data/sentiment_quantum'
    models_dir: str = 'models/quantum_operators'
    results_dir: str = 'results/quantum_intervention'

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if self.blend_ratios is None:
            self.blend_ratios = [0.02, 0.05, 0.1, 0.2]

    @property
    def memory_estimate_gb(self) -> float:
        """
        Estimate peak memory usage in GB

        Returns:
            Estimated peak memory in GB
        """
        # Unitary matrix: quantum_dim × quantum_dim complex (2 × float32 each)
        unitary_params = 2 * self.quantum_dim * self.quantum_dim
        unitary_memory = unitary_params * 4  # float32 = 4 bytes

        # Cayley transform workspace (matrix inversion)
        cayley_workspace = 3 * self.quantum_dim * self.quantum_dim * 8  # complex64 = 8 bytes

        # Training batch
        batch_memory = self.batch_size * self.quantum_dim * 8  # complex64

        # GPT-2 model (~500MB)
        gpt2_memory = 500 * 1024 * 1024

        # Total in bytes
        total_bytes = unitary_memory + cayley_workspace + batch_memory + gpt2_memory

        # Convert to GB
        return total_bytes / (1024 ** 3)

    @classmethod
    def local(cls) -> 'QuantumConfig':
        """
        Preset for local M1 MacBook (16GB RAM)

        Memory estimate: ~1.5GB peak
        Training time: ~10-20 min on M1
        Note: MPS doesn't support complex types, so we use CPU
        """
        return cls(
            quantum_dim=2000,
            learning_rate=0.001,
            epochs=100,
            batch_size=10,
            device='cpu'  # MPS doesn't support complex numbers
        )

    @classmethod
    def remote(cls) -> 'QuantumConfig':
        """
        Preset for cloud GPU (full scale)

        Memory estimate: ~6GB peak
        Training time: ~5-10 min on A4000
        """
        return cls(
            quantum_dim=10000,
            learning_rate=0.0005,
            epochs=150,
            batch_size=20,
            device='cuda'
        )

    @classmethod
    def tiny(cls) -> 'QuantumConfig':
        """
        Preset for rapid testing/debugging

        Memory estimate: ~200MB
        Training time: ~1 min
        """
        return cls(
            quantum_dim=500,
            learning_rate=0.001,
            epochs=50,
            batch_size=5,
            num_prompts=10,  # Only 10 prompts per class
            device='cpu'
        )

    @classmethod
    def from_preset(cls, preset: Literal['local', 'remote', 'tiny']) -> 'QuantumConfig':
        """
        Create config from preset name

        Args:
            preset: One of 'local', 'remote', 'tiny'

        Returns:
            QuantumConfig instance
        """
        if preset == 'local':
            return cls.local()
        elif preset == 'remote':
            return cls.remote()
        elif preset == 'tiny':
            return cls.tiny()
        else:
            raise ValueError(f"Unknown preset: {preset}. Choose 'local', 'remote', or 'tiny'")

    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("Quantum Configuration")
        print("=" * 70)
        print(f"  Quantum dimension:    {self.quantum_dim:,}-d")
        print(f"  Input dimension:      {self.input_dim}-d")
        print(f"  Training epochs:      {self.epochs}")
        print(f"  Batch size:           {self.batch_size}")
        print(f"  Learning rate:        {self.learning_rate}")
        print(f"  Prompts per class:    {self.num_prompts}")
        print(f"  Target layer:         {self.target_layer}")
        print(f"  Device:               {self.device}")
        print(f"  Blend ratios:         {self.blend_ratios}")
        print(f"  Estimated peak RAM:   {self.memory_estimate_gb:.2f} GB")
        print("=" * 70)


def get_device(device: str = 'auto'):
    """
    Get PyTorch device with auto-detection

    Args:
        device: 'auto', 'cpu', 'mps', or 'cuda'

    Returns:
        torch.device
    """
    import torch

    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)


def test_config():
    """Test configuration presets"""
    print("\nTesting Configuration Presets")
    print("=" * 70)

    presets = ['tiny', 'local', 'remote']

    for preset in presets:
        print(f"\n{preset.upper()} Preset:")
        config = QuantumConfig.from_preset(preset)
        config.print_summary()

        # Show comparison to classical HDC
        classical_dim = 10000
        if preset == 'local':
            print(f"\n  📊 Comparison to Classical HDC:")
            print(f"     Classical: {classical_dim:,}-d binary vectors")
            print(f"     Quantum:   {config.quantum_dim:,}-d complex vectors")
            print(f"     Information ratio: {config.quantum_dim * 2 / classical_dim:.1f}x")
            print(f"     (complex = 2 real values per dimension)")

        print()


if __name__ == "__main__":
    test_config()
