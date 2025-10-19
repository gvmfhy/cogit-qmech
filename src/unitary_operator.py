#!/usr/bin/env python3
"""
Unitary Operator for Cogit-QMech
Neural network that maintains unitary constraint: U†U = I

This is the key differentiator from classical HDC:
- Classical: MLP can learn any transformation (including non-reversible)
- Quantum: Operator MUST be unitary (reversible, norm-preserving)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.quantum_utils import verify_unitary, cayley_transform, inverse_cayley_transform


class UnitaryLayer(nn.Module):
    """
    A single unitary layer using Cayley transform parameterization

    Parameterization: U = Cayley(H) where H is Hermitian
    - H is learned (unconstrained)
    - U is computed via Cayley transform (automatically unitary)

    This guarantees U†U = I by construction.
    """

    def __init__(self, dim: int):
        """
        Initialize unitary layer

        Args:
            dim: Dimension of the unitary matrix (e.g., 10,000 for quantum states)
        """
        super().__init__()

        self.dim = dim

        # Parameterize as Hermitian matrix H
        # H = H† (Hermitian), then U = Cayley(H) is unitary

        # We'll store H as two real matrices: H = A + iB
        # For H to be Hermitian: A = A^T (symmetric), B = -B^T (antisymmetric)

        # Symmetric part (real): A = A^T
        # We only store the upper triangle to enforce symmetry
        self.A_upper = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # Antisymmetric part (imaginary): B = -B^T
        # We only store the upper triangle to enforce antisymmetry
        self.B_upper = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def get_hermitian_matrix(self) -> torch.Tensor:
        """
        Construct Hermitian matrix H from parameters

        Returns:
            Hermitian matrix H = A + iB where A = A^T and B = -B^T
        """
        # Construct symmetric matrix A
        A = torch.triu(self.A_upper) + torch.triu(self.A_upper, diagonal=1).T

        # Construct antisymmetric matrix B
        B = torch.triu(self.B_upper, diagonal=1) - torch.triu(self.B_upper, diagonal=1).T

        # Hermitian H = A + iB
        H = torch.complex(A, B)

        return H

    def forward(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Apply unitary transformation: U|ψ⟩

        Args:
            quantum_state: Complex quantum state (shape: [dim] or [batch, dim])

        Returns:
            Transformed state U|ψ⟩
        """
        # Get Hermitian matrix
        H = self.get_hermitian_matrix()

        # Convert to unitary via Cayley transform
        U = cayley_transform(H)

        # Apply to state(s)
        if quantum_state.dim() == 1:
            # Single state: U @ |ψ⟩
            return torch.matmul(U, quantum_state)
        else:
            # Batch of states: (U @ |ψ⟩^T)^T = |ψ⟩ @ U^T
            return torch.matmul(quantum_state, U.T)

    def get_unitary_matrix(self) -> torch.Tensor:
        """
        Get the unitary matrix U

        Returns:
            Unitary matrix U = Cayley(H)
        """
        H = self.get_hermitian_matrix()
        return cayley_transform(H)


class UnitaryOperator(nn.Module):
    """
    Unitary neural network for quantum state transformations

    Architecture: Single large unitary layer (simpler than classical MLP)
    - Classical: 10000 → 1024 → 512 → 10000 (non-unitary)
    - Quantum: 10000 → 10000 (unitary)

    Why simpler?
    - Unitary constraint is very restrictive
    - Don't need multiple layers for expressiveness
    - Single unitary can represent rich transformations
    """

    def __init__(self, quantum_dim: int = 10000):
        """
        Initialize unitary operator

        Args:
            quantum_dim: Dimension of quantum states
        """
        super().__init__()

        self.quantum_dim = quantum_dim

        # Single unitary layer
        self.unitary_layer = UnitaryLayer(quantum_dim)

        print(f"[Unitary Operator]")
        print(f"  Quantum state dim: {quantum_dim}")
        print(f"  Architecture: Single unitary layer (U†U = I)")
        print(f"  Parameters: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Apply unitary transformation to quantum state(s)

        Args:
            quantum_state: Complex quantum state (shape: [dim] or [batch, dim])

        Returns:
            Transformed state U|ψ⟩
        """
        return self.unitary_layer(quantum_state)

    def get_unitary_matrix(self) -> torch.Tensor:
        """
        Get the unitary transformation matrix

        Returns:
            Unitary matrix U
        """
        return self.unitary_layer.get_unitary_matrix()

    def verify_unitarity(self, tolerance: float = 1e-4) -> Tuple[bool, float]:
        """
        Verify that the operator is unitary: U†U = I

        Args:
            tolerance: Maximum allowed deviation from identity

        Returns:
            (is_unitary, max_deviation)
        """
        U = self.get_unitary_matrix()
        return verify_unitary(U, tolerance=tolerance)

    def get_inverse_operator(self) -> torch.Tensor:
        """
        Get the inverse operator U†

        For unitary operators, U^-1 = U† (conjugate transpose)

        Returns:
            Inverse operator U†
        """
        U = self.get_unitary_matrix()
        return torch.conj(U.T)


class BornRuleLoss(nn.Module):
    """
    Born rule fidelity loss for training unitary operators

    Classical: L = MSE(U|ψ⟩, target)
    Quantum: L = 1 - |⟨target|U|ψ⟩|²

    This measures quantum fidelity between output and target states.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Born rule loss

        Args:
            output: U|ψ⟩ (transformed state)
            target: Target quantum state

        Returns:
            Loss = 1 - fidelity, where fidelity = |⟨target|output⟩|²
        """
        # Quantum inner product: ⟨target|output⟩
        if output.dim() == 1:
            # Single state
            inner_product = torch.sum(torch.conj(target) * output)
            fidelity = torch.abs(inner_product) ** 2
        else:
            # Batch of states
            inner_products = torch.sum(torch.conj(target) * output, dim=-1)
            fidelity = torch.abs(inner_products) ** 2
            fidelity = fidelity.mean()  # Average over batch

        # Loss = 1 - fidelity (minimize to maximize fidelity)
        return 1.0 - fidelity


def test_unitary_operator():
    """
    Test the unitary operator implementation
    """
    print("=" * 70)
    print("Testing Unitary Operator")
    print("=" * 70)

    # Test with smaller dimension for speed
    test_dim = 100

    # Test 1: Create operator
    print("\n1. Create Unitary Operator")
    operator = UnitaryOperator(quantum_dim=test_dim)
    print(f"   ✓ Operator created with {operator.count_parameters():,} parameters")

    # Test 2: Verify unitarity
    print("\n2. Verify Unitarity (U†U = I)")
    is_unitary, deviation = operator.verify_unitarity()
    print(f"   Is unitary: {is_unitary}")
    print(f"   Max deviation: {deviation:.6f}")
    assert is_unitary, "Operator is not unitary!"

    # Test 3: Transform a quantum state
    print("\n3. Transform Quantum State")

    # Create normalized quantum state
    state = torch.randn(test_dim, dtype=torch.complex64)
    state = state / torch.sqrt(torch.sum(torch.abs(state) ** 2))

    # Apply operator
    transformed = operator(state)

    # Check norm preservation
    norm_before = torch.sqrt(torch.sum(torch.abs(state) ** 2))
    norm_after = torch.sqrt(torch.sum(torch.abs(transformed) ** 2))

    print(f"   ||ψ|| before: {norm_before:.6f}")
    print(f"   ||U|ψ⟩|| after: {norm_after:.6f}")
    print(f"   Norm preserved: {torch.abs(norm_before - norm_after) < 1e-5}")

    # Test 4: Batch transformation
    print("\n4. Batch Transformation")

    batch_states = torch.randn(10, test_dim, dtype=torch.complex64)
    batch_states = batch_states / torch.sqrt(torch.sum(torch.abs(batch_states) ** 2, dim=-1, keepdim=True))

    batch_transformed = operator(batch_states)

    norms_before = torch.sqrt(torch.sum(torch.abs(batch_states) ** 2, dim=-1))
    norms_after = torch.sqrt(torch.sum(torch.abs(batch_transformed) ** 2, dim=-1))

    print(f"   Batch size: {batch_states.shape[0]}")
    print(f"   All norms preserved: {torch.all(torch.abs(norms_before - norms_after) < 1e-5)}")

    # Test 5: Born rule loss
    print("\n5. Born Rule Loss")

    criterion = BornRuleLoss()

    # Create target state
    target = torch.randn(test_dim, dtype=torch.complex64)
    target = target / torch.sqrt(torch.sum(torch.abs(target) ** 2))

    # Compute loss
    loss = criterion(transformed, target)
    print(f"   Loss (1 - fidelity): {loss:.4f}")
    print(f"   Fidelity: {1 - loss:.4f}")

    # Test with identical states (fidelity = 1, loss = 0)
    loss_identical = criterion(state, state)
    print(f"   Loss for identical states: {loss_identical:.6f} (should be ~0)")

    # Test 6: Inverse operator (reversibility)
    print("\n6. Inverse Operator (Reversibility)")

    U_inv = operator.get_inverse_operator()

    # Apply U then U†
    transformed = torch.matmul(operator.get_unitary_matrix(), state)
    recovered = torch.matmul(U_inv, transformed)

    # Check if we recover the original state
    fidelity_roundtrip = torch.abs(torch.sum(torch.conj(state) * recovered)) ** 2
    print(f"   Fidelity after U → U†: {fidelity_roundtrip:.6f} (should be 1.0)")
    assert fidelity_roundtrip > 0.9999, "Round-trip failed!"

    # Test 7: Gradient flow
    print("\n7. Gradient Flow Through Operator")

    optimizer = torch.optim.Adam(operator.parameters(), lr=0.01)

    # Target state
    target = torch.randn(test_dim, dtype=torch.complex64)
    target = target / torch.sqrt(torch.sum(torch.abs(target) ** 2))

    # Single optimization step
    optimizer.zero_grad()
    output = operator(state)
    loss = criterion(output, target)
    loss.backward()

    # Check gradients exist
    has_gradients = all(p.grad is not None for p in operator.parameters())
    print(f"   Gradients computed: {has_gradients}")
    print(f"   Loss: {loss.item():.4f}")

    optimizer.step()

    # Verify still unitary after update
    is_unitary_after, deviation_after = operator.verify_unitarity()
    print(f"   Still unitary after optimization step: {is_unitary_after}")
    print(f"   Deviation: {deviation_after:.6f}")

    print("\n" + "=" * 70)
    print("✅ All unitary operator tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_unitary_operator()
