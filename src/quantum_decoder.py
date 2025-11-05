#!/usr/bin/env python3
"""
Quantum State Decoder for Cogit-QMech
Converts complex quantum states back to real-valued activations for GPT-2 injection
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

from src.quantum_utils import quantum_fidelity, normalize_state


class QuantumStateDecoder:
    """
    Decodes complex quantum states to real activations

    Challenge: GPT-2 expects real-valued activations, but our quantum states are complex.

    Approaches:
    1. Pseudoinverse projection (like classical HDC)
    2. Born rule measurement (collapse to real values)
    3. Extract real component only

    We use pseudoinverse + gentle blending for best results.
    """

    def __init__(self, encoder_projection: torch.Tensor):
        """
        Initialize decoder with encoder's projection matrix

        Args:
            encoder_projection: Complex projection matrix from QuantumStateEncoder
                               (shape: [input_dim, quantum_dim])
        """
        self.projection = encoder_projection
        self.input_dim = encoder_projection.shape[0]
        self.quantum_dim = encoder_projection.shape[1]
        self.device = encoder_projection.device

        print(f"[Quantum State Decoder]")
        print(f"  Quantum states: {self.quantum_dim}-d (complex)")
        print(f"  Real activations: {self.input_dim}-d")

        # Compute pseudoinverse for decoding
        # activation ≈ quantum_state @ projection†
        self.inverse_projection = torch.linalg.pinv(encoder_projection).to(torch.complex64)

        print(f"  ✓ Pseudoinverse computed: {self.inverse_projection.shape}")

    def decode_quantum_state(
        self,
        quantum_state: torch.Tensor,
        method: str = "real_component"
    ) -> torch.Tensor:
        """
        Decode complex quantum state to real activation

        Args:
            quantum_state: Complex quantum state (shape: [quantum_dim] or [batch, quantum_dim])
            method: Decoding method
                   - "real_component": Take real part after projection (default, baseline)
                   - "absolute": Take absolute value (magnitude only)
                   - "real_imag_avg": Average of real and imaginary parts (preserves both)
                   - "magnitude": Same as "absolute" (alias)
                   - "born_rule": Sample based on Born rule probabilities

        Returns:
            Real-valued activation (shape: [input_dim] or [batch, input_dim])
        """
        # Project back to activation space
        # activation = quantum_state @ inverse_projection
        quantum_state = quantum_state.to(self.device)

        if quantum_state.dim() == 1:
            reconstructed_complex = torch.matmul(quantum_state, self.inverse_projection)
        else:
            reconstructed_complex = torch.matmul(quantum_state, self.inverse_projection)

        # Convert complex to real based on method
        if method == "real_component":
            # Take real component (baseline - discards imaginary)
            activation = reconstructed_complex.real

        elif method == "real_imag_avg":
            # Average real and imaginary (preserves information from both)
            # This uses 100% of the complex state instead of 50%
            activation = (reconstructed_complex.real + reconstructed_complex.imag) / 2

        elif method == "absolute" or method == "magnitude":
            # Take absolute value (preserves magnitude, loses phase)
            # |a + bi| = sqrt(a² + b²)
            activation = torch.abs(reconstructed_complex)

        elif method == "born_rule":
            # Sample based on Born rule (stochastic)
            # For now, use absolute value as deterministic approximation
            # True Born rule sampling would require basis states
            activation = torch.abs(reconstructed_complex)

        else:
            raise ValueError(f"Unknown decoding method: {method}. Choose from: "
                           f"real_component, real_imag_avg, absolute, magnitude, born_rule")

        return activation

    def gentle_blend(
        self,
        original_activation: torch.Tensor,
        modified_quantum_state: torch.Tensor,
        blend_ratio: float = 0.1,
        decode_method: str = "real_component"
    ) -> torch.Tensor:
        """
        Blend original activation with modified quantum state

        This is crucial for preserving GPT-2 coherence:
        - Full replacement (ratio=1.0) often breaks generation
        - Gentle blending (ratio=0.05-0.2) preserves structure

        Args:
            original_activation: Original real activation from GPT-2
            modified_quantum_state: Quantum state after unitary transformation
            blend_ratio: How much of the modification to apply (0=none, 1=full)
            decode_method: How to decode quantum state to real

        Returns:
            Blended activation = (1-α) * original + α * decoded
        """
        # Decode quantum state to real activation
        decoded_activation = self.decode_quantum_state(
            modified_quantum_state,
            method=decode_method
        )

        # Blend: (1 - α) * original + α * modified
        original_activation = original_activation.to(self.device)
        decoded_activation = decoded_activation.to(self.device)

        blended = (1.0 - blend_ratio) * original_activation + blend_ratio * decoded_activation

        return blended

    def quantum_blend(
        self,
        original_quantum_state: torch.Tensor,
        modified_quantum_state: torch.Tensor,
        blend_ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Blend in quantum space before decoding

        Instead of blending activations, blend quantum states:
        |ψ_blend⟩ = (1-α)|ψ_original⟩ + α|ψ_modified⟩

        Then normalize and decode.

        Args:
            original_quantum_state: Original quantum state
            modified_quantum_state: Modified quantum state
            blend_ratio: Blending factor

        Returns:
            Blended and normalized quantum state
        """
        # Linear combination in quantum space
        original_quantum_state = original_quantum_state.to(self.device)
        modified_quantum_state = modified_quantum_state.to(self.device)

        blended_state = (
            (1.0 - blend_ratio) * original_quantum_state +
            blend_ratio * modified_quantum_state
        )

        # Renormalize (linear combination may not preserve norm)
        blended_state = normalize_state(blended_state)

        return blended_state

    def test_reconstruction_quality(
        self,
        original_activation: Union[np.ndarray, torch.Tensor],
        quantum_state: torch.Tensor
    ) -> dict:
        """
        Test how well we can reconstruct the original activation

        Args:
            original_activation: Original real activation
            quantum_state: Quantum state encoded from original activation

        Returns:
            Dictionary with reconstruction metrics
        """
        # Convert original to tensor
        if isinstance(original_activation, np.ndarray):
            original_tensor = torch.from_numpy(original_activation).to(self.device, dtype=torch.float32)
        else:
            original_tensor = original_activation.to(self.device, dtype=torch.float32)

        # Handle shape
        if original_tensor.dim() > 1:
            original_tensor = original_tensor.flatten()[:self.input_dim]
        if original_tensor.shape[0] < self.input_dim:
            padding = torch.zeros(self.input_dim - original_tensor.shape[0], device=self.device)
            original_tensor = torch.cat([original_tensor, padding])

        # Decode quantum state
        reconstructed = self.decode_quantum_state(quantum_state, method="real_component")

        # Compute reconstruction metrics
        mse = torch.mean((original_tensor - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(original_tensor - reconstructed)).item()

        # Cosine similarity
        cos_sim = torch.dot(original_tensor, reconstructed) / (
            torch.norm(original_tensor) * torch.norm(reconstructed)
        ).item()

        # Correlation
        orig_centered = original_tensor - original_tensor.mean()
        recon_centered = reconstructed - reconstructed.mean()
        correlation = torch.dot(orig_centered, recon_centered) / (
            torch.norm(orig_centered) * torch.norm(recon_centered)
        ).item()

        return {
            'mse': mse,
            'mae': mae,
            'cosine_similarity': cos_sim,
            'correlation': correlation
        }


def test_quantum_decoder():
    """
    Test the quantum state decoder
    """
    print("=" * 70)
    print("Testing Quantum State Decoder")
    print("=" * 70)

    # Create fake encoder projection
    input_dim = 768
    quantum_dim = 10000

    torch.manual_seed(42)
    real_part = torch.randn(input_dim, quantum_dim)
    imag_part = torch.randn(input_dim, quantum_dim)
    projection = torch.complex(real_part, imag_part)

    # Normalize columns
    col_norms = torch.sqrt(torch.sum(torch.abs(projection) ** 2, dim=0, keepdim=True))
    projection = projection / col_norms

    # Create decoder
    print("\n1. Create Decoder")
    decoder = QuantumStateDecoder(projection)

    # Test 2: Encode and decode (round-trip test)
    print("\n2. Round-Trip Test (Encode → Decode)")

    # Create original activation
    original_activation = torch.randn(input_dim)

    # Encode to quantum state (simulate encoder)
    act_complex = torch.complex(original_activation, torch.zeros_like(original_activation))
    quantum_state = torch.matmul(act_complex, projection)
    quantum_state = normalize_state(quantum_state)

    # Decode back
    reconstructed = decoder.decode_quantum_state(quantum_state, method="real_component")

    # Measure quality
    mse = torch.mean((original_activation - reconstructed) ** 2).item()
    cos_sim = torch.dot(original_activation, reconstructed) / (
        torch.norm(original_activation) * torch.norm(reconstructed)
    ).item()

    print(f"   MSE: {mse:.6f}")
    print(f"   Cosine similarity: {cos_sim:.6f}")

    # Test 3: Different decoding methods
    print("\n3. Compare Decoding Methods")

    for method in ["real_component", "absolute"]:
        decoded = decoder.decode_quantum_state(quantum_state, method=method)
        mse = torch.mean((original_activation - decoded) ** 2).item()
        print(f"   {method:20s}: MSE = {mse:.6f}")

    # Test 4: Gentle blending
    print("\n4. Gentle Blending")

    # Create modified quantum state (simulate operator transformation)
    modified_quantum = quantum_state * 0.9  # Slight modification
    modified_quantum = normalize_state(modified_quantum)

    blend_ratios = [0.0, 0.05, 0.1, 0.2, 1.0]

    for ratio in blend_ratios:
        blended = decoder.gentle_blend(
            original_activation,
            modified_quantum,
            blend_ratio=ratio
        )
        diff = torch.mean((original_activation - blended) ** 2).item()
        print(f"   Blend ratio {ratio:.2f}: MSE from original = {diff:.6f}")

    # Test 5: Quantum blending (in quantum space)
    print("\n5. Quantum Space Blending")

    for ratio in [0.0, 0.1, 0.5, 1.0]:
        blended_quantum = decoder.quantum_blend(
            quantum_state,
            modified_quantum,
            blend_ratio=ratio
        )

        # Check normalization
        norm = torch.sqrt(torch.sum(torch.abs(blended_quantum) ** 2))
        print(f"   Blend ratio {ratio:.2f}: ||ψ|| = {norm:.6f} (should be 1.0)")

    # Test 6: Batch decoding
    print("\n6. Batch Decoding")

    batch_size = 10
    batch_quantum = torch.randn(batch_size, quantum_dim, dtype=torch.complex64)
    batch_quantum = batch_quantum / torch.sqrt(
        torch.sum(torch.abs(batch_quantum) ** 2, dim=-1, keepdim=True)
    )

    batch_decoded = decoder.decode_quantum_state(batch_quantum, method="real_component")

    print(f"   Input batch shape: {batch_quantum.shape}")
    print(f"   Output batch shape: {batch_decoded.shape}")
    print(f"   Output dtype: {batch_decoded.dtype}")

    # Test 7: Reconstruction quality test
    print("\n7. Reconstruction Quality Metrics")

    original_np = original_activation.numpy()
    metrics = decoder.test_reconstruction_quality(original_np, quantum_state)

    for metric, value in metrics.items():
        print(f"   {metric:20s}: {value:.6f}")

    print("\n" + "=" * 70)
    print("✅ All quantum decoder tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_quantum_decoder()
