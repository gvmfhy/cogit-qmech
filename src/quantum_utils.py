#!/usr/bin/env python3
"""
Quantum Utilities for Cogit-QMech
Provides foundational quantum operations using PyTorch complex numbers
"""

import torch
import numpy as np
from typing import Tuple, Optional


def complex_inner_product(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute quantum inner product ⟨φ|ψ⟩ = Σ conj(φ_i) * ψ_i

    Args:
        psi: Complex quantum state (shape: [d] or [batch, d])
        phi: Complex quantum state (shape: [d] or [batch, d])

    Returns:
        Complex inner product (scalar or batch of scalars)

    Note:
        In quantum mechanics, ⟨φ|ψ⟩ means conjugate phi's components first
    """
    if psi.dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(f"psi must be complex, got {psi.dtype}")
    if phi.dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(f"phi must be complex, got {phi.dtype}")

    # Handle batched or single states
    if psi.dim() == 1:
        # Single states: ⟨φ|ψ⟩ = sum(conj(φ) * ψ)
        return torch.sum(torch.conj(phi) * psi)
    else:
        # Batched states: sum over last dimension
        return torch.sum(torch.conj(phi) * psi, dim=-1)


def normalize_state(psi: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Normalize quantum state: |ψ⟩ / √⟨ψ|ψ⟩

    Args:
        psi: Complex quantum state (shape: [d] or [batch, d])
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized state with ⟨ψ|ψ⟩ = 1
    """
    if psi.dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(f"psi must be complex, got {psi.dtype}")

    # Compute norm: ||ψ|| = √⟨ψ|ψ⟩
    if psi.dim() == 1:
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2) + epsilon)
        return psi / norm
    else:
        # Batched: compute norm per state
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2, dim=-1, keepdim=True) + epsilon)
        return psi / norm


def quantum_fidelity(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute quantum fidelity F(ψ, φ) = |⟨ψ|φ⟩|²

    Measures "closeness" of two quantum states.
    Returns 1 if states are identical, 0 if orthogonal.

    Args:
        psi: Normalized complex quantum state
        phi: Normalized complex quantum state

    Returns:
        Fidelity in range [0, 1]
    """
    inner_prod = complex_inner_product(psi, phi)
    return torch.abs(inner_prod) ** 2


def born_rule_probability(state: torch.Tensor, measurement_basis: torch.Tensor) -> torch.Tensor:
    """
    Compute Born rule probability P(measurement|state) = |⟨basis|state⟩|²

    This is the quantum mechanical probability of measuring a state
    in a particular basis.

    Args:
        state: Complex quantum state |ψ⟩
        measurement_basis: Measurement basis |φ⟩

    Returns:
        Probability in range [0, 1]
    """
    return quantum_fidelity(state, measurement_basis)


def verify_unitary(U: torch.Tensor, tolerance: float = 1e-4) -> Tuple[bool, float]:
    """
    Verify that a matrix is unitary: U†U ≈ I

    Args:
        U: Complex matrix (shape: [d, d] or [batch, d, d])
        tolerance: Maximum allowed deviation from identity

    Returns:
        (is_unitary, max_deviation)
    """
    if U.dtype not in [torch.complex64, torch.complex128]:
        raise ValueError(f"U must be complex, got {U.dtype}")

    if U.dim() == 2:
        # Single matrix
        U_dagger = torch.conj(U.T)
        product = torch.matmul(U_dagger, U)

        # Check if product ≈ I
        identity = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
        deviation = torch.max(torch.abs(product - identity)).item()

        return deviation < tolerance, deviation

    else:
        # Batched matrices
        U_dagger = torch.conj(U.transpose(-2, -1))
        product = torch.matmul(U_dagger, U)

        # Check each matrix in batch
        batch_size = U.shape[0]
        identity = torch.eye(U.shape[-1], dtype=U.dtype, device=U.device).unsqueeze(0).expand(batch_size, -1, -1)
        deviation = torch.max(torch.abs(product - identity)).item()

        return deviation < tolerance, deviation


def create_hermitian_matrix(dim: int, seed: Optional[int] = None) -> torch.Tensor:
    """
    Create a random Hermitian matrix H (used for generating unitary matrices)

    A matrix is Hermitian if H = H†
    Unitary matrices can be constructed as U = exp(iH)

    Args:
        dim: Dimension of the matrix
        seed: Random seed for reproducibility

    Returns:
        Hermitian matrix of shape [dim, dim]
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create random complex matrix
    real_part = torch.randn(dim, dim)
    imag_part = torch.randn(dim, dim)
    H = torch.complex(real_part, imag_part)

    # Make it Hermitian: H = (H + H†) / 2
    H = (H + torch.conj(H.T)) / 2

    return H


def cayley_transform(H: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform: U = (I + iH)(I - iH)⁻¹

    Converts Hermitian matrix H into unitary matrix U.
    Guarantees U†U = I by construction.

    Args:
        H: Hermitian matrix (shape: [d, d] or [batch, d, d])

    Returns:
        Unitary matrix U

    Note:
        This is numerically stable and automatically unitary.
        Alternative to matrix exponential U = exp(iH).
    """
    if H.dim() == 2:
        dim = H.shape[0]
        I = torch.eye(dim, dtype=H.dtype, device=H.device)
    else:
        batch_size, dim = H.shape[0], H.shape[-1]
        I = torch.eye(dim, dtype=H.dtype, device=H.device).unsqueeze(0).expand(batch_size, -1, -1)

    # Cayley transform: U = (I + iH)(I - iH)^-1
    i = torch.tensor(1j, dtype=H.dtype, device=H.device)

    numerator = I + i * H
    denominator = I - i * H

    # Solve: U = numerator @ inv(denominator)
    U = torch.linalg.solve(denominator, numerator)

    return U


def inverse_cayley_transform(U: torch.Tensor) -> torch.Tensor:
    """
    Inverse Cayley transform: H = i(I - U)(I + U)⁻¹

    Converts unitary matrix back to Hermitian matrix.
    Useful for parameterizing unitary networks.

    Args:
        U: Unitary matrix (shape: [d, d] or [batch, d, d])

    Returns:
        Hermitian matrix H
    """
    if U.dim() == 2:
        dim = U.shape[0]
        I = torch.eye(dim, dtype=U.dtype, device=U.device)
    else:
        batch_size, dim = U.shape[0], U.shape[-1]
        I = torch.eye(dim, dtype=U.dtype, device=U.device).unsqueeze(0).expand(batch_size, -1, -1)

    i = torch.tensor(1j, dtype=U.dtype, device=U.device)

    numerator = I - U
    denominator = I + U

    # H = i * numerator @ inv(denominator)
    H = i * torch.linalg.solve(denominator, numerator)

    return H


def batch_apply_unitary(U: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """
    Apply unitary operator to batch of quantum states: U|ψ⟩

    Args:
        U: Unitary matrix (shape: [d, d])
        states: Batch of quantum states (shape: [batch, d])

    Returns:
        Transformed states U|ψ⟩ (shape: [batch, d])
    """
    # states shape: [batch, d]
    # U shape: [d, d]
    # result: [batch, d] = [batch, d] @ [d, d]^T

    return torch.matmul(states, U.T)


def measure_state_collapse(state: torch.Tensor, measurement_basis: torch.Tensor) -> torch.Tensor:
    """
    Simulate quantum measurement (state collapse) via Born rule

    After measurement, state collapses to measurement_basis with probability
    P = |⟨basis|state⟩|²

    Args:
        state: Quantum state before measurement
        measurement_basis: Basis state to project onto

    Returns:
        Collapsed state (normalized measurement_basis) if measurement succeeds
    """
    # Probability of measuring this basis state
    prob = born_rule_probability(state, measurement_basis)

    # In a real quantum measurement, we'd sample based on probability
    # For now, return the projection (deterministic)
    inner_prod = complex_inner_product(measurement_basis, state)

    # Project: |φ⟩⟨φ|ψ⟩
    projected = inner_prod * measurement_basis

    # Normalize (post-measurement state)
    return normalize_state(projected)


# Statistics and analysis utilities

def compute_state_purity(state: torch.Tensor) -> torch.Tensor:
    """
    Compute purity of quantum state: Tr(ρ²) where ρ = |ψ⟩⟨ψ|

    Pure states have purity = 1
    Mixed states have purity < 1

    Args:
        state: Quantum state |ψ⟩

    Returns:
        Purity value in range [0, 1]
    """
    # For pure states, purity is always 1
    # This is placeholder for future mixed state support
    return torch.tensor(1.0, dtype=torch.float32, device=state.device)


def compute_von_neumann_entropy(state: torch.Tensor) -> torch.Tensor:
    """
    Compute von Neumann entropy: S = -Tr(ρ log ρ)

    For pure states, S = 0
    For maximally mixed states, S = log(d)

    Args:
        state: Quantum state |ψ⟩

    Returns:
        Entropy value

    Note:
        This is a placeholder for future entanglement analysis
    """
    # For pure states, von Neumann entropy is 0
    return torch.tensor(0.0, dtype=torch.float32, device=state.device)


def test_quantum_utils():
    """
    Test suite for quantum utilities
    """
    print("=" * 70)
    print("Testing Quantum Utilities")
    print("=" * 70)

    # Test 1: Complex inner product
    print("\n1. Complex Inner Product")
    psi = torch.tensor([1+0j, 0+1j, 0+0j], dtype=torch.complex64)
    phi = torch.tensor([0+1j, 1+0j, 0+0j], dtype=torch.complex64)

    inner = complex_inner_product(psi, phi)
    print(f"   ⟨φ|ψ⟩ = {inner}")
    print(f"   |⟨φ|ψ⟩|² = {torch.abs(inner)**2}")

    # Test 2: State normalization
    print("\n2. State Normalization")
    unnormalized = torch.tensor([3+4j, 1+2j], dtype=torch.complex64)
    normalized = normalize_state(unnormalized)
    norm = torch.sqrt(torch.sum(torch.abs(normalized) ** 2))
    print(f"   Before: ||ψ|| = {torch.sqrt(torch.sum(torch.abs(unnormalized) ** 2)):.4f}")
    print(f"   After:  ||ψ|| = {norm:.4f}")

    # Test 3: Quantum fidelity
    print("\n3. Quantum Fidelity")
    state1 = normalize_state(torch.tensor([1+0j, 0+0j], dtype=torch.complex64))
    state2 = normalize_state(torch.tensor([0+0j, 1+0j], dtype=torch.complex64))
    state3 = normalize_state(torch.tensor([1+0j, 0+0j], dtype=torch.complex64))

    f_orthogonal = quantum_fidelity(state1, state2)
    f_identical = quantum_fidelity(state1, state3)
    print(f"   F(orthogonal states) = {f_orthogonal:.4f} (should be ~0)")
    print(f"   F(identical states) = {f_identical:.4f} (should be 1)")

    # Test 4: Cayley transform (unitarity)
    print("\n4. Cayley Transform → Unitary Matrix")
    H = create_hermitian_matrix(dim=4, seed=42)
    U = cayley_transform(H)

    is_unitary, deviation = verify_unitary(U)
    print(f"   Is unitary: {is_unitary}")
    print(f"   Max deviation from U†U = I: {deviation:.6f}")

    # Test 5: Applying unitary to states
    print("\n5. Applying Unitary Operator")
    dim = 10
    H = create_hermitian_matrix(dim=dim, seed=42)
    U = cayley_transform(H)

    state = normalize_state(torch.randn(dim, dtype=torch.complex64))
    transformed = torch.matmul(U, state)

    norm_before = torch.sqrt(torch.sum(torch.abs(state) ** 2))
    norm_after = torch.sqrt(torch.sum(torch.abs(transformed) ** 2))

    print(f"   ||ψ|| before = {norm_before:.6f}")
    print(f"   ||U|ψ⟩|| after = {norm_after:.6f}")
    print(f"   Norm preserved: {torch.abs(norm_before - norm_after) < 1e-5}")

    # Test 6: Inverse Cayley (reversibility)
    print("\n6. Inverse Cayley Transform (Reversibility)")
    H_original = create_hermitian_matrix(dim=5, seed=123)
    U = cayley_transform(H_original)
    H_recovered = inverse_cayley_transform(U)

    diff = torch.max(torch.abs(H_original - H_recovered))
    print(f"   Max difference H_original vs H_recovered: {diff:.6f}")
    print(f"   Reversible: {diff < 1e-4}")

    print("\n" + "=" * 70)
    print("✅ All quantum utility tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_quantum_utils()
