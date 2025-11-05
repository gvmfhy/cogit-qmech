# Technical Debt Refactoring TODO

**Purpose:** Track code improvements needed before scaling to H100/A100 or production deployment.

**Priority Legend:**
- 🔴 **CRITICAL:** Breaks on H100/larger GPUs
- 🟡 **IMPORTANT:** Performance issue, but works
- 🟢 **NICE-TO-HAVE:** Code quality improvement

---

## ✅ COMPLETE: Device-Aware Operator Loading

**Status:** ✅ Implemented in Decision #003 refactoring (2025-11-04)

**Problem (was):** `experiments/sentiment/quantum_phase3_test.py` hardcoded operators to CPU

**Solution implemented (lines 81-108):**

**Implemented code:**
```python
def load_operator_smart(self, file_path, quantum_dim, device='cuda'):
    """Load operator to GPU if memory available, else CPU"""
    checkpoint = torch.load(file_path, map_location='cpu')
    operator = UnitaryOperator(quantum_dim)
    operator.load_state_dict(checkpoint['model_state_dict'])

    if device == 'cuda' and torch.cuda.is_available():
        # Check available GPU memory
        gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info()
        operator_bytes = sum(p.numel() * p.element_size() for p in operator.parameters())

        # 1.2× safety margin for fragmentation
        if gpu_free_bytes > operator_bytes * 1.2:
            operator.to(device)
            print(f"✓ Operator loaded to GPU ({operator_bytes/1e9:.2f} GB)")
            return operator, 'cuda'
        else:
            print(f"⚠️ Insufficient GPU memory ({gpu_free_bytes/1e9:.2f} GB free, need {operator_bytes/1e9:.2f} GB)")
            print(f"   Keeping operator on CPU (will be slower)")
            return operator, 'cpu'
    else:
        print(f"ℹ️ Operator on CPU (device={device})")
        return operator, 'cpu'
```

**Files to modify:**
- `experiments/sentiment/quantum_phase3_test.py:95-120`
- `experiments/sentiment/test_reversibility.py` (if it has same pattern)

**Test cases:**
1. RTX 5090 (32GB): Should use CPU (expected)
2. H100 (80GB): Should use GPU (critical!)
3. Mac M1 (unified memory): Should handle gracefully

**Validation:**
```python
# After loading
print(f"Operator device: {next(operator.parameters()).device}")
# Should be 'cuda:0' on H100, 'cpu' on RTX 5090
```

---

## 🔴 CRITICAL: Device-Aware Intervention Function

**Problem:** `quantum_intervention()` hardcodes CPU execution in line 179

**Current code:**
```python
def quantum_intervention(activations, hook):
    # Hardcoded to CPU
    quantum_state = encoder.encode_activation(activations.cpu().numpy())
    # Stays on CPU even if operator is on GPU!
    transformed_state = operator(quantum_state)
```

**Why it's broken:**
- If operator is on GPU (H100), quantum_state must also be on GPU
- Current code sends data to CPU → slow transfer → CPU compute → transfer back
- Wastes GPU completely

**Required fix:**
```python
def quantum_intervention(activations, hook):
    # Determine operator device
    operator_device = next(operator.parameters()).device

    # Encode (always happens on CPU first)
    quantum_state = encoder.encode_activation(activations.cpu().numpy())

    # Move to operator's device
    if operator_device.type == 'cuda':
        quantum_state = quantum_state.to(operator_device)

    # Apply operator (on whatever device it lives)
    with torch.no_grad():
        transformed_state = operator(quantum_state)

    # Blending happens on same device as operator
    if blend_ratio < 1.0:
        blended_state = decoder.quantum_blend(
            quantum_state,
            transformed_state,
            blend_ratio=blend_ratio
        )
    else:
        blended_state = transformed_state

    # Decode (move back to CPU for numpy conversion)
    decoded_activation = decoder.decode_quantum_state(
        blended_state.cpu(),
        method="real_component"
    )

    # ... rest of function
```

**Files to modify:**
- `experiments/sentiment/quantum_phase3_test.py:172-216`

---

## 🟡 IMPORTANT: GPU Memory Profiling

**Problem:** No automatic memory checking before loading models

**Current behavior:**
- Load model → crash with OOM
- User has to manually calculate if things fit

**Desired behavior:**
```python
def estimate_memory_requirements(config):
    """Calculate total memory needed before loading"""
    model_size = estimate_model_size(config.model_name)  # From HF model card
    operator_size = config.quantum_dim ** 2 * 2 * 4  # complex float32
    total_gb = (model_size + 2 * operator_size) / 1e9

    gpu_free, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)

    print(f"Memory Requirements:")
    print(f"  Model: {model_size/1e9:.2f} GB")
    print(f"  2× Operators: {2*operator_size/1e9:.2f} GB")
    print(f"  Total needed: {total_gb:.2f} GB")
    print(f"  GPU available: {gpu_free/1e9:.2f} GB / {gpu_total/1e9:.2f} GB")

    if total_gb > gpu_free / 1e9:
        print(f"⚠️ WARNING: Not enough GPU memory")
        print(f"   Recommend: Use quantization or keep operators on CPU")
        return False
    else:
        print(f"✓ Sufficient GPU memory")
        return True
```

**Call in Phase 3 __init__:**
```python
def __init__(self, config):
    if not estimate_memory_requirements(config):
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            raise MemoryError("Insufficient GPU memory")
```

---

## 🟡 IMPORTANT: Add GPU Memory Monitoring

**Problem:** Can't see memory usage during runs

**Desired feature:**
```python
def log_gpu_memory(stage=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        total_gb = total / 1e9

        print(f"[GPU Memory {stage}]")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Free:      {free_gb:.2f} / {total_gb:.2f} GB")
```

**Usage:**
```python
# In Phase 3
log_gpu_memory("After model load")
log_gpu_memory("After operators load")
log_gpu_memory("During generation")
```

---

## 🟢 NICE-TO-HAVE: Quantization Support

**Problem:** No built-in quantization option

**Feature request:**
```python
# config.py
@classmethod
def qwen_remote_q8(cls):
    """8-bit quantized Qwen for memory-constrained GPUs"""
    return cls(
        model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        quantization="int8",
        # ... rest same
    )
```

**Benefits:**
- 7B Q8: ~7GB vs 30GB FP16
- Allows operators on GPU even on RTX 5090
- 10-15× faster Phase 3

**Implementation:**
- Add quantization parameter to config
- Update TransformerLensAdapter to handle quantized models
- Test on multiple quantization libraries (GPTQ, AWQ, GGUF)

---

## 🟢 NICE-TO-HAVE: Automated H100 Setup Script

**Problem:** Manual setup on new pod is error-prone

**Feature request:**
`scripts/setup_h100.sh`
```bash
#!/bin/bash
# Automated H100 pod setup

# Clone repo
git clone https://github.com/gvmfhy/cogit-qmech.git
cd cogit-qmech

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Rsync data from backup
echo "Rsync quantum states and operators from your Mac:"
echo "  rsync -avz ~/cogit-qmech-backup/ /workspace/cogit-qmech/"

# Run full pipeline
python experiments/sentiment/quantum_phase3_test.py --preset qwen_remote
python experiments/sentiment/test_reversibility.py --preset qwen_remote
```

---

## Refactoring Checklist

**Before H200 migration:**
- [x] Implement `load_operator_smart()` with memory checking ✅ Decision #003
- [x] Update `quantum_intervention()` for device-aware execution ✅ Decision #003
- [x] Add GPU memory profiling to Phase 3 __init__ ✅ Decision #003
- [x] Add `log_gpu_memory()` calls throughout ✅ Decision #003
- [x] Test on Mac (CPU fallback) ✅ Decision #003
- [x] Document changes in DECISIONS.md ✅ Decision #003, #004

**After H200 deployment:**
- [x] Validate operators load to GPU ✅ qwen_remote: 0.70 GB each on CUDA
- [x] Measure Phase 3 speedup (should be ~10×) ✅ ~3 min on H200 vs projected 2hrs CPU
- [x] Fix device auto-detection bug ✅ Decision #004 (all presets now use device='auto')
- [ ] Add quantization support for Q8 (future work)
- [ ] Create automated setup script (future work)

**Validation Results (H200 SXM, 143GB VRAM):**
- qwen_tiny (1500-d): ✅ All phases complete, operators on GPU (0.02 GB each)
- qwen_remote (9333-d): ✅ All phases complete, operators on GPU (0.70 GB each)
- GPU memory usage: 32.22 GB / 150 GB (21.5% utilization)
- Intervention quality: ✅ Clear sentiment shifts observed
- Reversibility: ✅ 0.986 fidelity (excellent)

---

**Last Updated:** 2025-11-05 01:35 UTC
**Status:** ✅ COMPLETE - All critical refactoring validated on H200
