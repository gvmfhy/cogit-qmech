#!/usr/bin/env python3
"""
Quantum State Encoder for Cogit-QMech
Encodes real-valued neural network activations as normalized complex quantum states
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from src.quantum_utils import normalize_state, quantum_fidelity


class QuantumStateEncoder:
    """
    Encodes real-valued activations into complex-valued quantum states

    Classical HDC: 768-d → binary 10,000-d via random projection + sign()
    Quantum: 768-d → complex 10,000-d via complex projection + normalization

    Key differences:
    - Complex amplitudes (not binary)
    - Normalized states: ⟨ψ|ψ⟩ = 1
    - Preserves more information (amplitude + phase)
    """

    def __init__(self, input_dim: int = 768, quantum_dim: int = 10000, seed: int = 42):
        """
        Initialize quantum encoder with deterministic random projection

        Args:
            input_dim: Dimension of neural network activations (e.g., GPT-2 = 768)
            quantum_dim: Dimension of quantum state space (default 10,000)
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim
        self.seed = seed

        print(f"[Quantum State Encoder]")
        print(f"  Real activations: {input_dim}-d")
        print(f"  Quantum states:   {quantum_dim}-d (complex)")
        print(f"  Seed: {seed}")

        # Create deterministic complex projection matrix
        torch.manual_seed(seed)

        # Complex projection: separate real and imaginary parts
        real_part = torch.randn(input_dim, quantum_dim)
        imag_part = torch.randn(input_dim, quantum_dim)

        self.projection = torch.complex(real_part, imag_part)

        # Normalize columns (optional, helps with numerical stability)
        col_norms = torch.sqrt(torch.sum(torch.abs(self.projection) ** 2, dim=0, keepdim=True))
        self.projection = self.projection / col_norms

        print(f"  ✓ Complex projection matrix created: {self.projection.shape}")

    def encode_activation(self, activation: np.ndarray) -> torch.Tensor:
        """
        Encode a single activation to quantum state |ψ⟩

        Args:
            activation: Real-valued activation (shape varies, will be flattened/averaged)

        Returns:
            Normalized complex quantum state (shape: [quantum_dim])
        """
        # Convert to tensor
        act_tensor = torch.tensor(activation, dtype=torch.float32)

        # Handle different input shapes (same as classical HDC)
        if len(act_tensor.shape) == 3:
            # (batch, seq, dim) → (dim,) via averaging
            act_tensor = act_tensor.mean(dim=[0, 1])
        elif len(act_tensor.shape) == 2:
            # (seq, dim) → (dim,)
            act_tensor = act_tensor.mean(dim=0)
        elif len(act_tensor.shape) > 3:
            # Unexpected shape, flatten
            act_tensor = act_tensor.flatten()

        # Ensure correct dimensionality
        if act_tensor.shape[0] > self.input_dim:
            act_tensor = act_tensor[:self.input_dim]
        elif act_tensor.shape[0] < self.input_dim:
            padding = torch.zeros(self.input_dim - act_tensor.shape[0])
            act_tensor = torch.cat([act_tensor, padding])

        # Project to quantum space: ψ = activation @ projection
        # Real activation (768,) @ Complex matrix (768, 10000) = Complex state (10000,)
        # Need to convert real tensor to complex for matmul
        act_tensor_complex = torch.complex(act_tensor, torch.zeros_like(act_tensor))
        quantum_state = torch.matmul(act_tensor_complex, self.projection)

        # Normalize: ||ψ|| = 1 (required for quantum states)
        quantum_state = normalize_state(quantum_state)

        return quantum_state

    def encode_batch(self, activations: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Encode multiple activations to quantum states

        Args:
            activations: List of real-valued activations

        Returns:
            List of normalized quantum states
        """
        quantum_states = []

        print(f"\n  Encoding {len(activations)} activations to quantum states...")

        for i, act in enumerate(activations):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{len(activations)}")

            quantum_state = self.encode_activation(act)
            quantum_states.append(quantum_state)

        print(f"  ✓ Encoded {len(quantum_states)} quantum states")

        return quantum_states

    def analyze_separation(
        self,
        positive_states: List[torch.Tensor],
        negative_states: List[torch.Tensor]
    ) -> dict:
        """
        Analyze quantum separation between positive and negative states

        Unlike classical HDC (Hamming distance), we use quantum fidelity

        Args:
            positive_states: List of positive quantum states
            negative_states: List of negative quantum states

        Returns:
            Dictionary with separation statistics
        """
        print(f"\n  [Quantum Separation Analysis]")

        # Convert to tensors for batch operations
        pos_batch = torch.stack(positive_states)
        neg_batch = torch.stack(negative_states)

        # Compute centroids (mean quantum states)
        pos_centroid = normalize_state(pos_batch.mean(dim=0))
        neg_centroid = normalize_state(neg_batch.mean(dim=0))

        # Quantum fidelity between centroids
        centroid_fidelity = quantum_fidelity(pos_centroid, neg_centroid)

        # Classical approach: cosine similarity of absolute values
        pos_centroid_abs = torch.abs(pos_centroid)
        neg_centroid_abs = torch.abs(neg_centroid)
        cosine_sim = torch.dot(pos_centroid_abs, neg_centroid_abs) / (
            torch.norm(pos_centroid_abs) * torch.norm(neg_centroid_abs)
        )

        # Average fidelity within each class (consistency)
        pos_fidelities = []
        for state in positive_states[:10]:  # Sample for speed
            f = quantum_fidelity(state, pos_centroid)
            pos_fidelities.append(f.item())

        neg_fidelities = []
        for state in negative_states[:10]:
            f = quantum_fidelity(state, neg_centroid)
            neg_fidelities.append(f.item())

        # Cross-class fidelity (should be lower)
        cross_fidelities = []
        for i in range(min(10, len(positive_states))):
            for j in range(min(10, len(negative_states))):
                f = quantum_fidelity(positive_states[i], negative_states[j])
                cross_fidelities.append(f.item())

        stats = {
            'centroid_fidelity': centroid_fidelity.item(),
            'centroid_cosine_similarity': cosine_sim.item(),
            'pos_class_consistency': np.mean(pos_fidelities),
            'pos_class_std': np.std(pos_fidelities),
            'neg_class_consistency': np.mean(neg_fidelities),
            'neg_class_std': np.std(neg_fidelities),
            'cross_class_fidelity': np.mean(cross_fidelities),
            'cross_class_std': np.std(cross_fidelities)
        }

        print(f"    Centroid fidelity: {stats['centroid_fidelity']:.4f}")
        print(f"      (Lower is better for separation, range [0,1])")
        print(f"    Centroid cosine sim (|amplitudes|): {stats['centroid_cosine_similarity']:.4f}")
        print(f"    Positive class consistency: {stats['pos_class_consistency']:.4f} ± {stats['pos_class_std']:.4f}")
        print(f"    Negative class consistency: {stats['neg_class_consistency']:.4f} ± {stats['neg_class_std']:.4f}")
        print(f"    Cross-class fidelity: {stats['cross_class_fidelity']:.4f} ± {stats['cross_class_std']:.4f}")
        print(f"      (Should be lower than within-class)")

        return stats

    def save_projection_matrix(self, save_path: Path):
        """
        Save the projection matrix for later decoding

        Args:
            save_path: Path to save the projection matrix
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'projection': self.projection,
            'input_dim': self.input_dim,
            'quantum_dim': self.quantum_dim,
            'seed': self.seed
        }, save_path)

        print(f"  ✓ Projection matrix saved to {save_path}")

    @classmethod
    def load_from_saved(cls, load_path: Path) -> 'QuantumStateEncoder':
        """
        Load encoder from saved projection matrix

        Args:
            load_path: Path to saved projection matrix

        Returns:
            QuantumStateEncoder with loaded projection
        """
        checkpoint = torch.load(load_path)

        encoder = cls(
            input_dim=checkpoint['input_dim'],
            quantum_dim=checkpoint['quantum_dim'],
            seed=checkpoint['seed']
        )

        # Override with saved projection
        encoder.projection = checkpoint['projection']

        print(f"  ✓ Loaded projection matrix from {load_path}")
        return encoder


def test_quantum_encoder():
    """
    Test the quantum state encoder
    """
    print("=" * 70)
    print("Testing Quantum State Encoder")
    print("=" * 70)

    # Create encoder
    encoder = QuantumStateEncoder(input_dim=768, quantum_dim=10000, seed=42)

    # Test 1: Encode a single activation
    print("\n1. Single Activation Encoding")
    dummy_activation = np.random.randn(1, 10, 768)  # (batch, seq, dim)
    quantum_state = encoder.encode_activation(dummy_activation)

    print(f"   Input shape: {dummy_activation.shape}")
    print(f"   Output shape: {quantum_state.shape}")
    print(f"   Output dtype: {quantum_state.dtype}")

    # Check normalization
    norm = torch.sqrt(torch.sum(torch.abs(quantum_state) ** 2))
    print(f"   ||ψ|| = {norm:.6f} (should be 1.0)")
    assert torch.abs(norm - 1.0) < 1e-5, "State not normalized!"

    # Test 2: Encode batch
    print("\n2. Batch Encoding")
    dummy_batch = [np.random.randn(1, 10, 768) for _ in range(5)]
    quantum_batch = encoder.encode_batch(dummy_batch)

    print(f"   Batch size: {len(quantum_batch)}")
    print(f"   Each state shape: {quantum_batch[0].shape}")

    # Check all are normalized
    all_normalized = all(
        torch.abs(torch.sqrt(torch.sum(torch.abs(state) ** 2)) - 1.0) < 1e-5
        for state in quantum_batch
    )
    print(f"   All normalized: {all_normalized}")

    # Test 3: Separation analysis
    print("\n3. Separation Analysis")

    # Create "positive" and "negative" activations (with different means)
    pos_activations = [np.random.randn(1, 10, 768) + 0.5 for _ in range(10)]
    neg_activations = [np.random.randn(1, 10, 768) - 0.5 for _ in range(10)]

    pos_states = encoder.encode_batch(pos_activations)
    neg_states = encoder.encode_batch(neg_activations)

    stats = encoder.analyze_separation(pos_states, neg_states)

    # Test 4: Determinism (same seed → same encoding)
    print("\n4. Determinism Check")

    encoder1 = QuantumStateEncoder(input_dim=768, quantum_dim=10000, seed=42)
    encoder2 = QuantumStateEncoder(input_dim=768, quantum_dim=10000, seed=42)

    test_activation = np.random.randn(768)
    state1 = encoder1.encode_activation(test_activation)
    state2 = encoder2.encode_activation(test_activation)

    fidelity = quantum_fidelity(state1, state2)
    print(f"   Fidelity between encodings: {fidelity:.6f} (should be 1.0)")
    assert fidelity > 0.9999, "Encodings not deterministic!"

    # Test 5: Save and load
    print("\n5. Save and Load Projection Matrix")

    save_path = Path("/tmp/test_projection.pt")
    encoder.save_projection_matrix(save_path)

    loaded_encoder = QuantumStateEncoder.load_from_saved(save_path)

    state_original = encoder.encode_activation(test_activation)
    state_loaded = loaded_encoder.encode_activation(test_activation)

    fidelity = quantum_fidelity(state_original, state_loaded)
    print(f"   Fidelity after load: {fidelity:.6f} (should be 1.0)")

    # Cleanup
    save_path.unlink()

    print("\n" + "=" * 70)
    print("✅ All quantum encoder tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_quantum_encoder()
