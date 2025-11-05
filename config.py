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
    GPT-2 (124M params, 768-d hidden):
    - 'local': Optimized for M1 MacBook 16GB (quantum_dim=2000)
    - 'remote': Full-scale for cloud GPU (quantum_dim=10000)
    - 'tiny': Fast testing/debugging (quantum_dim=500)

    Qwen2.5-3B (3.09B params, 2048-d hidden):
    - 'qwen_local': Optimized for M1 MacBook 16GB (quantum_dim=5333)
    - 'qwen_tiny': Fast testing/debugging (quantum_dim=1500)
    - 'qwen_test_layers': Test multiple layers

    Qwen2.5-7B (7.61B params, 3584-d hidden):
    - 'qwen_remote': Full-scale for cloud GPU (quantum_dim=9333)
    """

    # Model selection
    model_name: str = "gpt2"      # Model to use (e.g., "gpt2", "Qwen/Qwen2.5-3B-Instruct")

    # Model dimensions
    input_dim: int = 768          # Model activation dimension (auto-detected if using TransformerLens)
    quantum_dim: int = 2000       # Quantum state dimension

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 10
    epochs: int = 100

    # Data collection
    num_prompts: int = 50         # Per sentiment class
    target_layer: int = 6         # Layer to extract activations from

    # Layer testing (for finding optimal layer)
    test_layers: list = None      # Layers to test (None = use single target_layer)

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
    def model_identifier(self) -> str:
        """
        Generate a short identifier for the model (for filenames)

        Returns:
            Short model name like 'gpt2', 'qwen2.5-3B'
        """
        if "Qwen2.5-3B" in self.model_name:
            return "qwen2.5-3B"
        elif "Qwen2.5-7B" in self.model_name:
            return "qwen2.5-7B"
        elif "gemma-2b" in self.model_name.lower():
            return "gemma-2b"
        elif "phi-3" in self.model_name.lower():
            return "phi-3"
        elif self.model_name == "gpt2":
            return "gpt2"
        else:
            # Generic fallback: take last part after / and limit length
            return self.model_name.split("/")[-1][:20].replace("/", "-")

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
            device='auto'  # Auto-detect GPU, fallback to CPU (MPS not supported)
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
            device='auto'  # Auto-detect GPU, fallback to CPU
        )

    @classmethod
    def tiny(cls) -> 'QuantumConfig':
        """
        Preset for rapid testing/debugging (GPT-2)

        Memory estimate: ~200MB
        Training time: ~1 min
        """
        return cls(
            model_name="gpt2",
            input_dim=768,
            quantum_dim=500,
            learning_rate=0.001,
            epochs=50,
            batch_size=5,
            num_prompts=10,  # Only 10 prompts per class
            target_layer=6,
            device='auto'  # Auto-detect GPU, fallback to CPU
        )

    @classmethod
    def qwen_local(cls) -> 'QuantumConfig':
        """
        Preset for Qwen2.5-3B on M1 MacBook (16GB RAM)

        Model: 3.09B params, 2048-d hidden, 36 layers
        Quantum: 5333-d (proportional to GPT-2 ratio)
        Memory estimate: ~3.8GB peak
        Training time: ~15-25 min on M1
        """
        return cls(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            input_dim=2048,
            quantum_dim=5333,
            learning_rate=0.001,
            epochs=100,
            batch_size=10,
            num_prompts=50,
            target_layer=18,  # ~50% depth for richer semantics
            device='auto'  # Auto-detect GPU, fallback to CPU
        )

    @classmethod
    def qwen_tiny(cls) -> 'QuantumConfig':
        """
        Preset for rapid testing/debugging with Qwen2.5-3B

        Memory estimate: ~800MB
        Training time: ~3-5 min
        """
        return cls(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            input_dim=2048,
            quantum_dim=1500,
            learning_rate=0.001,
            epochs=50,
            batch_size=5,
            num_prompts=10,
            target_layer=18,
            device='auto'  # Auto-detect GPU, fallback to CPU
        )

    @classmethod
    def qwen_test_layers(cls) -> 'QuantumConfig':
        """
        Preset for testing multiple layers with Qwen2.5-3B

        Tests layers at different depths to find optimal layer
        Memory estimate: ~1.5GB
        """
        return cls(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            input_dim=2048,
            quantum_dim=2000,
            learning_rate=0.001,
            epochs=50,
            batch_size=5,
            num_prompts=20,
            target_layer=18,  # Default if test_layers not specified
            test_layers=[6, 12, 18, 24, 30],  # Test at 17%, 33%, 50%, 67%, 83% depth
            device='auto'  # Auto-detect GPU, fallback to CPU
        )

    @classmethod
    def qwen_remote(cls) -> 'QuantumConfig':
        """
        Preset for Qwen2.5-7B on remote GPU (DigitalOcean/RunPod)

        Model: 7.61B params, 3584-d hidden, 28 layers
        Quantum: 9333-d (proportional to GPT-2 ratio: 3584 * 2.604)
        Memory estimate: ~8.5GB peak (needs GPU with 12GB+ VRAM)
        Training time: ~3-5 min on RTX 4090
        Device: CUDA required
        """
        return cls(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            input_dim=3584,
            quantum_dim=9333,
            learning_rate=0.0008,
            epochs=100,
            batch_size=16,
            num_prompts=50,
            target_layer=14,  # ~50% depth (28 layers total)
            device='auto'  # Auto-detect GPU, fallback to CPU if needed
        )

    @classmethod
    def from_preset(cls, preset: Literal['local', 'remote', 'tiny', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote']) -> 'QuantumConfig':
        """
        Create config from preset name

        Args:
            preset: One of 'local', 'remote', 'tiny', 'qwen_local', 'qwen_tiny', 'qwen_test_layers'

        Returns:
            QuantumConfig instance
        """
        preset_map = {
            'local': cls.local,
            'remote': cls.remote,
            'tiny': cls.tiny,
            'qwen_local': cls.qwen_local,
            'qwen_tiny': cls.qwen_tiny,
            'qwen_test_layers': cls.qwen_test_layers,
            'qwen_remote': cls.qwen_remote,
        }

        if preset in preset_map:
            return preset_map[preset]()
        else:
            valid_presets = ', '.join(preset_map.keys())
            raise ValueError(f"Unknown preset: {preset}. Choose from: {valid_presets}")

    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("Quantum Configuration")
        print("=" * 70)
        print(f"  Model:                {self.model_name}")
        print(f"  Model ID:             {self.model_identifier}")
        print(f"  Input dimension:      {self.input_dim}-d")
        print(f"  Quantum dimension:    {self.quantum_dim:,}-d")
        print(f"  Target layer:         {self.target_layer}")
        if self.test_layers:
            print(f"  Test layers:          {self.test_layers}")
        print(f"  Training epochs:      {self.epochs}")
        print(f"  Batch size:           {self.batch_size}")
        print(f"  Learning rate:        {self.learning_rate}")
        print(f"  Prompts per class:    {self.num_prompts}")
        print(f"  Device:               {self.device}")
        print(f"  Blend ratios:         {self.blend_ratios}")
        print(f"  Estimated peak RAM:   {self.memory_estimate_gb:.2f} GB")
        print("=" * 70)


def get_device(device: str = 'auto', verbose: bool = True):
    """
    Get PyTorch device with auto-detection and optional GPU info

    Args:
        device: 'auto', 'cpu', 'mps', or 'cuda'
        verbose: If True, print device information

    Returns:
        torch.device
    """
    import torch

    if device == 'auto':
        if torch.cuda.is_available():
            selected_device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            selected_device = torch.device('mps')
        else:
            selected_device = torch.device('cpu')
    else:
        selected_device = torch.device(device)

    # Print device info if verbose
    if verbose:
        print(f"\n[Device Information]")
        print(f"  Selected device: {selected_device}")

        if selected_device.type == 'cuda':
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print(f"  ⚠️  CUDA requested but not available! Falling back to CPU.")
                selected_device = torch.device('cpu')

        elif selected_device.type == 'mps':
            if torch.backends.mps.is_available():
                print(f"  Apple Metal Performance Shaders (MPS) available")
            else:
                print(f"  ⚠️  MPS requested but not available! Falling back to CPU.")
                selected_device = torch.device('cpu')

        elif selected_device.type == 'cpu':
            import os
            cpu_count = os.cpu_count()
            print(f"  CPU cores: {cpu_count}")
            print(f"  Note: Complex number ops require CPU (MPS doesn't support them)")

    return selected_device


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
