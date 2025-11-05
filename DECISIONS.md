# Strategic Decision Log: Cogit-QMech Experiments

**Purpose:** This document serves as a "Captain's Log" recording strategic decisions made during experiments. It documents not just *what* we did, but *why* we did it, what alternatives were considered, and what tradeoffs were accepted.

**Format:** Each decision includes full context, analysis, options considered, and reasoning. This creates a searchable record to help future researchers understand why the codebase and experimental approach evolved the way it did.

---

## Decision #001: GPU Memory Crisis - CPU Fallback Strategy

**Date:** 2025-11-04
**Time:** ~00:15 UTC
**Context:** Running Phase 3 (intervention testing) on RunPod RTX 5090 (32GB) with Qwen2.5-7B-Instruct

### The Problem

Phase 3 crashed with `torch.OutOfMemoryError` when attempting to load unitary operators to GPU alongside the 7B model.

**Memory calculation:**
```
Qwen 7B model (FP16):     ~30 GB  (loaded on GPU for inference)
Operator U_pos→neg:       ~1.4 GB (174M params × 2 complex × 4 bytes)
Operator U_neg→pos:       ~1.4 GB
----------------------------------------
Total needed:             ~32.8 GB
GPU available:            32 GB (RTX 5090)
```

**Why Phase 2 worked but Phase 3 failed:**
- **Phase 2:** Only operators on GPU, no language model → fits easily
- **Phase 3:** Model + operators simultaneously → doesn't fit

### Initial Mistake

First attempted to "fix" GPU utilization by moving operators to GPU:

```python
# experiments/sentiment/quantum_phase3_test.py:103, 119
self.operator_pos_to_neg.to(device)  # WRONG - causes OOM
```

This failed immediately with OOM error.

---

### Full Strategic Analysis

#### Current Phase 3 Timing Estimate (CPU Operators)

**What Phase 3 does:**
- 5 test prompts
- 2 intervention directions (pos→neg, neg→pos)
- 4 blend ratios each
- 1 baseline per prompt
- **Total: 5 baseline + 40 interventions = 45 generations**

**Current bottleneck:**
- Operators (174M params each) on CPU doing complex matrix multiplication
- Each generation requires:
  1. Model forward pass (GPU) - fast
  2. Extract activation (GPU→CPU) - fast
  3. Encode to quantum state (CPU) - fast
  4. **Apply operator (CPU matmul 9333×9333 complex) - SLOW**
  5. Decode (CPU) - fast
  6. Return to GPU - fast

**Estimated time per generation:** ~2-3 minutes (based on CPU matmul of 174M params)

**Total Phase 3 time estimate: ~90-135 minutes (1.5-2.25 hours)**

This is **unacceptably slow** for iterative experimentation.

---

#### H100 Performance Comparison

**RTX 5090 specs:**
- FP16 TFLOPS: ~165
- Memory bandwidth: ~1TB/s
- VRAM: 32GB

**H100 specs:**
- FP16 TFLOPS: ~1,979 (12× faster compute)
- Memory bandwidth: ~3.35TB/s (3.4× faster)
- VRAM: 80GB (2.5× more)

**On H100 with operators on GPU:**
- Model + operators would fit easily (30GB + 3GB = 33GB < 80GB)
- GPU matmul ~10-15× faster than CPU
- **Estimated Phase 3 time: ~5-10 minutes** (vs 90-135 min on RTX 5090 w/ CPU operators)

**On RTX 5090 with quantized model + GPU operators:**
- Q8 model (~7GB) + operators (~3GB) = 10GB < 32GB ✅
- GPU matmul ~10-15× faster than CPU
- **Estimated Phase 3 time: ~6-12 minutes** (vs 90-135 min current)

---

#### Code Changes: Will They Scale or Hurt Us?

**Changes made so far:**

##### 1. CPU operator loading (`experiments/sentiment/quantum_phase3_test.py`)

```python
checkpoint = torch.load(file, map_location='cpu')  # Not 'cuda'
operator.load_state_dict(checkpoint)
# No .to(device) call
```

**Impact on scaling:**
- ❌ **ANTI-PATTERN for larger GPUs**
- This change was a workaround for 32GB limit
- On H100 (80GB) or with quantization, we'd want to REVERT this
- **Should be device-aware:**

```python
# Better approach:
if has_gpu_space_for_operators():
    operator.to(device)
```

##### 2. Hybrid CPU/GPU execution in intervention function

```python
quantum_state = encoder.encode_activation(activations.cpu().numpy())
# Keep on CPU - operator is on CPU
transformed_state = operator(quantum_state)  # Runs on CPU
```

**Impact on scaling:**
- ⚠️ **CONDITIONALLY HARMFUL**
- Hardcoded to CPU execution
- On H100, this would still use CPU unnecessarily
- **Needs conditional logic:**

```python
# Better:
if operator.device.type == 'cuda':
    quantum_state = quantum_state.to(device)
    transformed_state = operator(quantum_state)  # GPU
else:
    quantum_state = quantum_state.cpu()
    transformed_state = operator(quantum_state)  # CPU fallback
```

##### 3. Documentation in TROUBLESHOOTING.md

**Impact:** ✅ **POSITIVE**
- Explains *why* we made these choices
- Future us (or researchers) will understand constraints

---

#### What We're Learning (That Will Scale)

**Good patterns emerging:**
1. **Memory calculation methodology** - applies to any hardware
2. **Phase separation** (1: data, 2: training, 3: testing) - scales well
3. **Checkpointing strategy** - critical for long runs
4. **Git-based workflow** - essential for multi-GPU experiments

**Bad patterns (technical debt):**
1. **Hardcoded CPU fallbacks** - should be dynamic based on available memory
2. **No memory profiling in code** - should check `torch.cuda.memory_available()` before loading
3. **TransformerLens device handling** - not tested with quantization

---

### Options Considered

#### A) Let it complete (2 hours)
- ✅ Get baseline results
- ✅ Validate end-to-end pipeline
- ❌ Wastes time if we then redo with quantization
- ❌ Costs ~$1.78 for slow results

#### B) Kill and switch to quantized model now
- ✅ Faster iterations (6-12 min vs 2 hours)
- ✅ Better use of GPU
- ✅ Establishes pattern for future runs
- ❌ Adds ~30 min to set up quantization
- ❌ Risk if quantization breaks something

#### C) Let this complete, then add quantization as Run 002
- ✅ Have both FP16 and Q8 baselines for comparison
- ✅ Less risky
- ❌ Slower tonight
- ✅ Better science (can measure quantization impact)

---

### Decision Made

**Option C: Let FP16 complete, add Q8 quantization as Run 002**

### Reasoning

1. **Already invested:** 1+ hour into Phase 3 - components loaded successfully
2. **Scientific value:** Having FP16 baseline validates that everything works end-to-end
3. **Enables comparison:** Can measure FP16 vs Q8 quality differences scientifically
4. **Lower risk:** Don't break working code mid-experiment
5. **Sufficient time:** ~6.5 hours left on RunPod ($6.59 remaining @ $0.89/hr) for both runs

### Tradeoffs Accepted

- **Speed:** Accept 2-hour Phase 3 run for FP16 baseline
- **Cost:** ~$1.78 for slower inference vs ~$0.20 for Q8 (but gain scientific comparison)
- **Iteration time:** Slower experimentation tonight, but establishes both baselines

### Technical Debt Created

**Files modified with CPU workarounds:**
- `experiments/sentiment/quantum_phase3_test.py:98` - `map_location='cpu'`
- `experiments/sentiment/quantum_phase3_test.py:103` - No `.to(device)` call
- `experiments/sentiment/quantum_phase3_test.py:115` - `map_location='cpu'`
- `experiments/sentiment/quantum_phase3_test.py:119` - No `.to(device)` call
- `experiments/sentiment/quantum_phase3_test.py:179` - CPU-only quantum state processing

**Refactoring needed for H100/A100:**
```python
# Add device-aware operator loading
def load_operator_smart(file_path, quantum_dim, device='cuda'):
    """Load operator to GPU if memory available, else CPU"""
    checkpoint = torch.load(file_path, map_location='cpu')
    operator = UnitaryOperator(quantum_dim)
    operator.load_state_dict(checkpoint['model_state_dict'])

    # Try GPU first
    if device == 'cuda':
        gpu_free = torch.cuda.mem_get_info()[0]
        operator_size = sum(p.numel() * p.element_size() for p in operator.parameters())

        if gpu_free > operator_size * 1.2:  # 20% safety margin
            operator.to(device)
            print(f"✓ Operator loaded to GPU ({operator_size/1e9:.2f} GB)")
        else:
            print(f"⚠️ Insufficient GPU memory, keeping operator on CPU")

    return operator
```

### Validation Criteria

**How we'll know this was the right decision:**
1. **FP16 Run 001 completes successfully** - validates pipeline
2. **Q8 Run 002 runs 10× faster** - validates quantization approach
3. **Quality comparison shows <2% degradation** - acceptable for Q8
4. **Both runs documented in EXPERIMENT_LOG.md** - scientific record preserved

### Future Implications

**For 70B model experiments:**
- Q8 quantization becomes mandatory (70B FP16 = ~140GB, won't fit on H100)
- This decision establishes the quantization workflow
- CPU fallback code provides safety net for memory-constrained setups

**For production deployment:**
- Q8 provides 10× speedup with minimal quality loss
- Worth the engineering effort to implement properly

---

## Decision Template (for future entries)

```markdown
## Decision #XXX: [Title]

**Date:** YYYY-MM-DD
**Time:** HH:MM UTC
**Context:** [What was happening when this decision was needed]

### The Problem
[Describe the issue that forced a decision]

### Full Strategic Analysis
[Complete reasoning, calculations, benchmarks]

### Options Considered
#### A) [Option 1]
- ✅ Pros
- ❌ Cons

#### B) [Option 2]
- ✅ Pros
- ❌ Cons

### Decision Made
[What we chose]

### Reasoning
[Why we chose it - numbered list]

### Tradeoffs Accepted
[What we sacrificed]

### Technical Debt Created
[Code that needs refactoring later]

### Validation Criteria
[How we'll know if it was right]

### Future Implications
[Impact on future work]
```

---

**Last Updated:** 2025-11-04
**Decisions Logged:** 1
