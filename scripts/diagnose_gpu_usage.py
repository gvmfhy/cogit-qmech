#!/usr/bin/env python3
"""
Diagnose GPU usage in quantum encoding pipeline
Check where tensors actually live and if complex matmul uses GPU
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from src.quantum_encoder import QuantumStateEncoder

print("=" * 70)
print("GPU USAGE DIAGNOSTIC")
print("=" * 70)

# Check CUDA availability
print("\n[1] CUDA Availability:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  PyTorch version: {torch.__version__}")

# Create encoder
print("\n[2] Creating Encoder:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = QuantumStateEncoder(input_dim=1024, quantum_dim=2666, seed=42, device=device)

# Check projection matrix device
print("\n[3] Projection Matrix Device:")
print(f"  projection.device: {encoder.projection.device}")
print(f"  projection.dtype: {encoder.projection.dtype}")
print(f"  projection.is_cuda: {encoder.projection.is_cuda}")
print(f"  projection.real.device: {encoder.projection.real.device}")
print(f"  projection.imag.device: {encoder.projection.imag.device}")

# Test encoding with device tracking
print("\n[4] Testing Encoding with Device Tracking:")

# Create dummy activation
dummy_activation = np.random.randn(1, 10, 1024)

# Monkey-patch to track device during encoding
original_matmul = torch.matmul

def tracked_matmul(a, b):
    print(f"\n  [matmul called]")
    print(f"    Input A: device={a.device}, dtype={a.dtype}, shape={a.shape}, is_cuda={a.is_cuda}")
    print(f"    Input B: device={b.device}, dtype={b.dtype}, shape={b.shape}, is_cuda={b.is_cuda}")
    
    # Time the operation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
    result = original_matmul(a, b)
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        print(f"    Time: {elapsed_ms:.3f} ms")
    
    print(f"    Output: device={result.device}, dtype={result.dtype}, shape={result.shape}, is_cuda={result.is_cuda}")
    
    return result

# Temporarily replace matmul
torch.matmul = tracked_matmul

try:
    quantum_state = encoder.encode_activation(dummy_activation)
    print(f"\n[5] Final Quantum State:")
    print(f"  device: {quantum_state.device}")
    print(f"  dtype: {quantum_state.dtype}")
    print(f"  is_cuda: {quantum_state.is_cuda}")
    print(f"  shape: {quantum_state.shape}")
finally:
    # Restore original matmul
    torch.matmul = original_matmul

# Test if complex matmul actually uses GPU
print("\n[6] Direct Complex Matmul Test:")
if torch.cuda.is_available():
    # Create complex tensors on GPU
    a_real = torch.randn(1024, device='cuda')
    a_imag = torch.zeros_like(a_real)
    a_complex = torch.complex(a_real, a_imag)
    
    b_real = torch.randn(1024, 2666, device='cuda')
    b_imag = torch.randn(1024, 2666, device='cuda')
    b_complex = torch.complex(b_real, b_imag)
    
    print(f"  a_complex: device={a_complex.device}, is_cuda={a_complex.is_cuda}")
    print(f"  b_complex: device={b_complex.device}, is_cuda={b_complex.is_cuda}")
    
    # Time complex matmul
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result = torch.matmul(a_complex, b_complex)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    
    print(f"  result: device={result.device}, is_cuda={result.is_cuda}")
    print(f"  Time: {elapsed_ms:.3f} ms")
    
    # Compare to decomposed version
    print("\n[7] Decomposed Real Matmul Test:")
    torch.cuda.synchronize()
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    
    start2.record()
    real_part = torch.matmul(a_real, b_real)
    imag_part = torch.matmul(a_real, b_imag)
    result2 = torch.complex(real_part, imag_part)
    end2.record()
    torch.cuda.synchronize()
    
    elapsed_ms2 = start2.elapsed_time(end2)
    
    print(f"  Time: {elapsed_ms2:.3f} ms")
    print(f"  Speedup: {elapsed_ms / elapsed_ms2:.2f}x")
    
    # Check if results are identical
    print(f"\n[8] Results Match:")
    print(f"  Max difference: {torch.max(torch.abs(result - result2)).item():.2e}")
    print(f"  Results identical: {torch.allclose(result, result2, atol=1e-6)}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

