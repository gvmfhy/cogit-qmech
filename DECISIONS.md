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

## Decision #002: Abort RTX 5090, Migrate to H100

**Date:** 2025-11-04
**Time:** ~00:30 UTC
**Context:** Phase 3 running on RTX 5090 with CPU operators (estimated 2 hours to complete)

### The Problem

Phase 1 & 2 completed successfully on RTX 5090, but Phase 3 revealed critical performance issue:

**Measured performance:**
- Operators on CPU due to memory constraints
- First prompt baseline: Generated successfully
- Estimated time for full Phase 3: **90-135 minutes** (unacceptably slow)
- Each of 45 generations: ~2-3 minutes (CPU matrix multiplication bottleneck)

**Why slow:**
- CPU matmul for 174M parameter operators (9333×9333 complex matrices)
- Data transfers: GPU → CPU → CPU compute → CPU → GPU
- No parallelization on CPU vs GPU tensor cores

### Data Already Saved

**Completed & portable work:**
- ✅ Phase 1: Quantum states (41MB, saved to Mac)
- ✅ Phase 2: 2× operators trained to 98.9% fidelity (2.6GB, saved to Mac)
- ✅ Documentation: TROUBLESHOOTING.md, DECISIONS.md, TODO_REFACTORING.md
- ✅ Code improvements: Device handling, git workflow

**Total rsync:**Transfer time: ~3 minutes

**What we're abandoning:**
- ❌ Phase 3 partial run (~30 min invested, ~90-120 min remaining)
- Decision: Sunk cost, not worth waiting

---

### Options Considered

#### A) Wait for RTX 5090 Phase 3 to complete (2 hours)
- ✅ Get FP16 baseline results
- ✅ Validate full pipeline
- ❌ Costs ~$1.78 ($0.89/hr × 2hr)
- ❌ Wastes time (can't iterate)
- ❌ No time left for Phase 4 or Run 002

#### B) Kill Phase 3, switch to quantization on RTX 5090
- ✅ Faster (Q8: 6-12 min vs 2 hours)
- ✅ Operators fit on GPU
- ❌ 30-60 min setup time (find Q8 model, test compatibility)
- ❌ Risk: TransformerLens may not support quantization
- ❌ Still limited to 32GB for future 70B experiments

#### C) Kill Phase 3, migrate to H100 (80GB)
- ✅ **10-15× faster** (GPU operators: 5-10 min vs 2 hours)
- ✅ All data portable (Phases 1 & 2 reusable)
- ✅ Validates proper GPU execution path
- ✅ Establishes workflow for 70B experiments
- ✅ Total time: 20 min (Phase 3: 10 min + Phase 4: 5 min + setup: 5 min)
- ❌ Costs ~$0.50-$1.00 (20-40 min @ ~$1.50/hr typical H100 rate)
- ⚠️ Requires pod availability

---

### Decision Made

**Option C: Abort RTX 5090, migrate to H100**

### Reasoning

1. **Sunk cost fallacy avoided:** 30 min invested < 90 min remaining
2. **Science quality:** H100 run validates operators work properly on GPU (important for paper)
3. **Time efficiency:** H100 completes Phases 3+4 in 20 min vs 2+ hours on RTX 5090
4. **Cost efficiency:** ~$0.50 H100 vs $1.78 RTX 5090
5. **Future-proofing:** Establishes H100 workflow needed for 70B experiments
6. **Learning opportunity:** Tests portability of Phase 1 & 2 artifacts

### Tradeoffs Accepted

- **No FP16 CPU baseline:** We won't have slow-mode results to compare
  - Mitigation: Not scientifically valuable anyway (CPU is just a workaround)
- **Pod availability risk:** H100 pods may not be available immediately
  - Mitigation: Try multiple providers (RunPod, Lambda Labs, Vast.ai)
- **Setup time:** 5-10 min to configure new pod
  - Mitigation: Rsync already complete, just need env setup

### Actions Taken

1. ✅ Killed Phase 3 on RTX 5090 (pkill -9)
2. ✅ Rsynced data to Mac:
   - `~/cogit-qmech-backup/data/sentiment_quantum/` (558MB)
   - `~/cogit-qmech-backup/models/quantum_operators/` (2.6GB)
3. ✅ Created TODO_REFACTORING.md documenting technical debt
4. 📝 Next: Shut down RTX 5090 pod, spin up H100

### Technical Debt Acknowledged

**Files with hardcoded CPU that will break on H100:**
- `experiments/sentiment/quantum_phase3_test.py:95-120` - Force CPU loading
- `experiments/sentiment/quantum_phase3_test.py:179` - Force CPU execution

**Critical refactoring needed before H100:**
- Device-aware operator loading with memory checking
- Device-aware intervention function
- GPU memory profiling
- See TODO_REFACTORING.md for details

**Plan:** Refactor ASAP after H100 pod is up

### Validation Criteria

**How we'll know this was the right decision:**
1. **H100 Phase 3 completes in <15 min** (vs 2hr projected on RTX 5090)
2. **Operators load to GPU successfully** (validates refactoring works)
3. **Results match expectations** (sentiment shifts visible)
4. **Phase 4 completes** (have time for reversibility testing)
5. **Total cost < $1.00** (cheaper than waiting on RTX 5090)

### Future Implications

**For 70B experiments:**
- This migration validates data portability
- H100 workflow established (clone → rsync → run)
- Proves Phase 1 & 2 artifacts are model-independent

**For production:**
- Demonstrates value of device-aware code
- Shows importance of memory profiling
- Validates hybrid CPU/GPU as last resort, not design

---

**Last Updated:** 2025-11-04 00:30 UTC
**Decisions Logged:** 2
