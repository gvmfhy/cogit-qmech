#!/usr/bin/env python3
"""Simple GPU check for complex tensor operations"""

import torch

print("=" * 70)
print("SIMPLE GPU CHECK")
print("=" * 70)

# Check CUDA
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   PyTorch: {torch.__version__}")

# Test complex tensor creation on GPU
print(f"\n2. Creating complex tensors on GPU:")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    real_part = torch.randn(1024, 2666, device=device, dtype=torch.float32)
    imag_part = torch.randn(1024, 2666, device=device, dtype=torch.float32)
    
    print(f"   real_part.device: {real_part.device}")
    print(f"   imag_part.device: {imag_part.device}")
    
    # Create complex tensor
    complex_tensor = torch.complex(real_part, imag_part)
    
    print(f"   complex_tensor.device: {complex_tensor.device}")
    print(f"   complex_tensor.dtype: {complex_tensor.dtype}")
    print(f"   complex_tensor.is_cuda: {complex_tensor.is_cuda}")
    
    # Test .to() operation
    complex_tensor_64 = complex_tensor.to(torch.complex64)
    print(f"   After .to(complex64):")
    print(f"     device: {complex_tensor_64.device}")
    print(f"     is_cuda: {complex_tensor_64.is_cuda}")
    
except Exception as e:
    print(f"   ERROR: {e}")

# Test complex matmul
print(f"\n3. Testing complex matmul:")
try:
    if torch.cuda.is_available():
        a = torch.randn(1024, device='cuda', dtype=torch.float32)
        proj_real = torch.randn(1024, 2666, device='cuda', dtype=torch.float32)
        proj_imag = torch.randn(1024, 2666, device='cuda', dtype=torch.float32)
        proj_complex = torch.complex(proj_real, proj_imag).to(torch.complex64)
        
        print(f"   a.device: {a.device}")
        print(f"   proj_complex.device: {proj_complex.device}")
        
        a_complex = torch.complex(a, torch.zeros_like(a))
        print(f"   a_complex.device: {a_complex.device}")
        
        # This is the critical operation
        result = torch.matmul(a_complex, proj_complex)
        
        print(f"   result.device: {result.device}")
        print(f"   result.is_cuda: {result.is_cuda}")
        print(f"   ✅ Complex matmul stayed on GPU!")
        
    else:
        print(f"   Skipping (no CUDA)")
        
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 70)

