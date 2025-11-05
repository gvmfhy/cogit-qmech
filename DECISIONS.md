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

## Decision #003: Model-Agnostic Refactoring & Progressive Pipeline

**Date:** 2025-11-04
**Time:** ~01:30 UTC
**Context:** Preparing codebase for H100 deployment after RTX 5090 migration decision

### The Problem

Codebase had grown organically with several issues hindering scalability:

1. **Model-specific naming:** 10+ references to "GPT-2" in code meant for any language model
2. **Hardcoded device handling:** CPU hardcoded in Phase 1, inconsistent auto-detection elsewhere
3. **Technical debt from RTX 5090:** CPU operator fallbacks that would break on H100
4. **No systematic testing:** Developers running phases manually, risking errors
5. **Lack of validation:** No automated checks between phases

**Examples of model-specific code:**
```python
# quantum_phase1_collect.py:4
"""Collect GPT-2 activations and encode as quantum states"""

# quantum_phase3_test.py:39
class QuantumInterventionSystem:
    """Test quantum interventions on GPT-2"""

# quantum_phase3_test.py:74
def load_gpt2(self):  # Should be load_model()
```

**Risk:** Code appears GPT-2 specific even though it's model-agnostic. Confuses future researchers and makes Qwen2.5-7B work seem like an afterthought.

---

### Full Strategic Analysis

#### What Makes an Experiment Robust?

Based on lessons from RTX 5090 session:

1. **Fail fast, fail loud:** Catch bugs in 2-minute tiny run before 2-hour qwen_remote run
2. **Independent experiments:** Each preset generates its own data (different models = different distributions)
3. **Validation between phases:** Don't waste GPU time if Phase 2 fidelity is <85%
4. **Timestamp checking:** Prevent "using stale data" bugs (documented in TROUBLESHOOTING.md)
5. **Auto-logging:** Append to EXPERIMENT_LOG.md on success (creates scientific record)

#### Progressive Testing Strategy

Instead of meta-orchestration (auto-run tiny → qwen_remote), use **single-preset pipelines:**

```bash
# Validation run (fast, cheap)
python run_full_pipeline.py --preset tiny  # ~2-3 min

# Production run (slow, expensive) - only if tiny passed
python run_full_pipeline.py --preset qwen_remote  # ~20-30 min
```

**Why not auto-chain?**
- Each scale is an independent experiment deserving its own EXPERIMENT_LOG entry
- Researcher should inspect tiny results before committing to expensive runs
- Explicit control over when to spend money

---

### Options Considered

#### A) Fix only critical device bugs, skip refactoring
- ✅ Faster (30 min vs 2 hrs)
- ❌ Leaves confusing "GPT-2" references everywhere
- ❌ No systematic testing workflow
- ❌ Technical debt compounds

#### B) Comprehensive refactoring + progressive pipeline
- ✅ Clean, model-agnostic codebase
- ✅ Systematic validation reduces bugs
- ✅ Better for paper (shows generality)
- ✅ Easier onboarding for future researchers
- ❌ Takes 1-2 hours upfront
- ❌ Risk of introducing bugs during refactoring

#### C) Create pipeline only, leave naming as-is
- ✅ Get systematic testing quickly
- ❌ Still confusing for Qwen experiments
- ❌ Half-measure

---

### Decision Made

**Option B: Comprehensive refactoring + progressive pipeline**

### Reasoning

1. **Experimental robustness:** Progressive pipeline with validation prevents expensive failures
2. **Scientific clarity:** Model-agnostic code shows framework applies to any LLM
3. **Cost savings:** Catching bugs in 2-min tiny runs saves $1-2 per avoided qwen_remote failure
4. **H100 readiness:** Device-aware code critical for 80GB deployment
5. **Paper quality:** Demonstrates generality beyond GPT-2
6. **Time investment pays off:** 2 hrs now saves 10+ hrs debugging over next week

### Tradeoffs Accepted

- **Time:** 1-2 hours refactoring vs 30 min quick fix
  - Mitigation: Work is local on Mac, doesn't consume GPU credits
- **Regression risk:** Refactoring might introduce bugs
  - Mitigation: Test with `--preset tiny` before deployment

---

### Changes Made

#### Part 1: Model-Agnostic Refactoring

**quantum_phase1_collect.py:**
- Line 4: "Collect GPT-2 activations" → "Collect language model activations"
- Line 55: Removed hardcoded `device = 'cpu'`, use `config.device` with auto-detection
- Line 129: "Extract GPT-2 activations" → "Extract language model activations"
- Line 303: Updated argparse help for clarity

**quantum_phase2_train.py:**
- Line 416: "Test interventions on GPT-2" → "Test interventions on language model"
- Line 427: Updated argparse help

**quantum_phase3_test.py:**
- Line 4: "Apply operators to GPT-2" → "Apply operators to language model generation"
- Line 39: "Test interventions on GPT-2" → "Test interventions on language models"
- Line 74: `def load_gpt2()` → `def load_model()`
- Line 65: Updated call to `self.load_model()`
- Line 79: Use `config.device` instead of auto-detect (consistency with Phase 1)
- Line 81-120: Added `load_operator_smart()` with device-aware memory checking
- Line 210-257: Updated `quantum_intervention()` to be device-aware
- Line 41-53: Added `log_gpu_memory()` for tracking

**Device-aware operator loading:**
```python
def load_operator_smart(self, file_path, quantum_dim, device='cuda'):
    """Load operator to GPU if memory available, else CPU"""
    checkpoint = torch.load(file_path, map_location='cpu')
    operator = UnitaryOperator(quantum_dim)
    operator.load_state_dict(checkpoint['model_state_dict'])

    if device == 'cuda' and torch.cuda.is_available():
        gpu_free_bytes, _ = torch.cuda.mem_get_info()
        operator_bytes = sum(p.numel() * p.element_size() for p in operator.parameters())

        if gpu_free_bytes > operator_bytes * 1.2:  # 20% safety margin
            operator.to(device)
            return operator, 'cuda'
    return operator, 'cpu'
```

#### Part 2: Progressive Pipeline

**New file: `experiments/sentiment/run_full_pipeline.py`**

Features:
- Accepts `--preset` argument (tiny, qwen_tiny, qwen_remote, etc.)
- Runs Phases 1→2→3→4 sequentially with validation:
  - Phase 1: Check quantum states generated
  - Phase 2: Verify fidelity > 0.85 (fail if not)
  - Phase 3: Check results JSON created
  - Phase 4: Check reversibility results
- Data freshness handling:
  - Check if Phase 1 data exists
  - If < 1 hour old: prompt "Reuse or regenerate?"
  - If stale or missing: run Phase 1
- Tracks timing for each phase
- Calculates total cost (for remote presets)
- Auto-appends to EXPERIMENT_LOG.md on success (TODO)
- Fails fast on any error (no partial results)

**Example usage:**
```bash
# Fast validation (2-3 min, free on Mac)
python experiments/sentiment/run_full_pipeline.py --preset tiny

# Production run (20-30 min, ~$0.50 on H100)
python experiments/sentiment/run_full_pipeline.py --preset qwen_remote
```

---

### Technical Debt Resolved

**From TODO_REFACTORING.md:**
- ✅ Device-aware operator loading (Phase 3)
- ✅ Device-aware intervention function (Phase 3)
- ✅ GPU memory profiling (log_gpu_memory)
- ✅ Model-agnostic naming (all phases)
- ✅ Consistent device handling (all phases)

**Remaining debt:**
- ⏳ Test on actual H100 to validate GPU operator loading
- ⏳ Add EXPERIMENT_LOG.md auto-append functionality

---

### Validation Criteria

**How we'll know this was the right decision:**

1. **Local `--preset tiny` passes all 4 phases** (validates no regressions)
2. **H100 operators load to GPU** (validates device-aware code works)
3. **Phase times logged accurately** (enables cost/performance analysis)
4. **Code reads cleanly** (no GPT-2 confusion for Qwen experiments)
5. **Future experiments use pipeline** (proves it's useful)

### Future Implications

**For 70B experiments:**
- Progressive pipeline catches compatibility issues early
- Device-aware code handles >140GB model requirements
- Model-agnostic naming makes adding new models trivial

**For paper writing:**
- Clean code shows framework generality
- Progressive testing demonstrates robustness
- Timing data provides performance comparisons

**For collaboration:**
- New researchers can understand codebase faster
- Pipeline provides standardized experimental workflow
- Documentation shows decision-making process

---

## Decision #004: H200 Deployment & Device Auto-Detection Fix

**Date:** 2025-11-05
**Time:** ~01:00 UTC
**Context:** Deploying refactored code to RunPod H200 SXM (143GB VRAM) for full-scale Qwen2.5-7B testing

### The Problem

After completing Decision #003 refactoring, attempted to deploy and run `qwen_tiny` experiment on H200. **Critical bug discovered:** Models were loading on CPU despite 141GB of available GPU memory.

**User observation (paraphrased):**
> "Let's pause. Why are models being loaded on cpu not gpu?"
> "I do worry that fall back to cpu is a red flag"
> "It is also a sign that you were mistaken on your refactor"

**Output showing bug:**
```
Device:               cpu
✓ Model loaded on cpu
```

### Root Cause Analysis

**What went wrong:**
1. Decision #003 refactored phase scripts to respect `config.device`
2. Added device auto-detection logic in phase scripts (lines 55-66)
3. **BUT:** Forgot to update config preset definitions in `config.py`
4. All 7 presets still hardcoded `device='cpu'` or `device='cuda'`

**Why this broke auto-detection:**
```python
# config.py line 210 (qwen_tiny preset)
device='cpu'  # ❌ HARDCODED - bypasses auto-detection!

# Phase script (quantum_phase1_collect.py lines 55-66)
if config.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config.device  # ❌ Uses 'cpu' from preset
```

The auto-detection logic was present but never executed because presets specified explicit devices.

### Options Considered

#### A) Keep current behavior, document as "feature"
- ✅ No code changes needed
- ✅ Allows users to force CPU for debugging
- ❌ Silent failures (user doesn't know GPU was available)
- ❌ Violates principle of least surprise
- ❌ Makes expensive GPU hardware useless by default
- ❌ User explicitly worried this was a "red flag"

#### B) Change presets to use device='auto'
- ✅ Works everywhere (Mac, RTX 5090, H100, H200)
- ✅ Automatically uses GPU when available
- ✅ Still falls back to CPU gracefully
- ✅ Users can override via CLI args if needed
- ✅ Matches user expectation
- ❌ Need to update 7 preset methods

#### C) Remove device parameter entirely, always auto-detect
- ✅ Simplest user experience
- ❌ Removes explicit device control
- ❌ Makes debugging harder (can't force CPU)
- ❌ Breaks backward compatibility

---

### Decision Made

**Option B: Change all config presets to device='auto'**

### Reasoning

1. **User expectation:** When paying $1.50/hr for H200, expect GPU usage by default
2. **Fail-safe design:** Auto-detection + logging makes failures visible
3. **Portability:** Same preset works on Mac (CPU), RTX 5090 (32GB), H200 (143GB)
4. **Override-able:** Users can still force device via `config.device = 'cpu'` if needed
5. **Learning from failure:** User was right to be concerned - this WAS a bug

### Implementation

**Files modified:**
- `config.py` lines 132, 148, 168, 190, 210, 231, 254

**Changes:**
```python
# Before (ALL 7 presets):
device='cpu'      # or device='cuda'

# After (ALL 7 presets):
device='auto'     # Auto-detect GPU, fallback to CPU
```

**Presets updated:**
1. `local()` - line 132
2. `remote()` - line 148
3. `tiny()` - line 168
4. `qwen_local()` - line 190
5. `qwen_tiny()` - line 210 (the one that revealed the bug)
6. `qwen_test_layers()` - line 231
7. `qwen_remote()` - line 254

### Validation Results

**Test 1: qwen_tiny on H200 (after fix)**
```
Device:               auto
✓ Model loaded on cuda

[GPU Memory After model load]
  Allocated: 11.48 GB
  Reserved:  11.56 GB
  Free:      137.42 / 150.02 GB

Loading U_pos→neg (1,500-d):
  → Operator loaded to GPU (0.02 GB)
✓ Loaded U_pos→neg on cuda
```

**Test 2: qwen_remote (full 7B) on H200**
```
Device:               auto
✓ Model loaded on cuda

[GPU Memory After model load]
  Allocated: 30.79 GB
  Reserved:  30.85 GB
  Free:      118.62 / 150.02 GB

Loading U_pos→neg (9,333-d):
  → Operator loaded to GPU (0.70 GB)
✓ Loaded U_pos→neg on cuda

Loading U_neg→pos (9,333-d):
  → Operator loaded to GPU (0.70 GB)
✓ Loaded U_neg→pos on cuda
```

**Phase timings (H200, qwen_remote):**
- Phase 1: ~90 seconds (model load + data collection)
- Phase 2: ~402 seconds (training 2 operators, 100 epochs each)
- Phase 3: ~240 seconds (intervention testing, 6 prompts × 9 conditions)

**Total GPU memory:** 32.22 GB / 150 GB (21.5% utilization, excellent headroom)

### Intervention Quality Check

**Sample results show clear sentiment shifts:**

Prompt: "The project manager announced that"
- **Baseline**: "the project is 35% complete and will be finished within budget"
- **U_pos→neg (0.02)**: "the project team will be terminated at the end of the month" ✓ Negative shift!
- **U_neg→pos (0.05)**: "the project is moving into the next phase and will have a kick-off meeting" ✓ Positive shift!

**Technical validation:**
- ✅ Both operators maintain unitarity (deviation <0.00003)
- ✅ Reversibility: ~0.986 (excellent)
- ✅ Final fidelities: 0.983 (pos→neg), 0.997 (neg→pos)
- ✅ All operators on GPU (not CPU fallback)

### What We Learned

**About refactoring:**
1. **Check the whole call chain** - Refactored phase scripts but forgot config presets
2. **User testing is critical** - User caught the bug immediately by observing output
3. **Silent failures are dangerous** - CPU fallback without warning would waste expensive GPU time
4. **Document assumptions** - Should have verified all presets during Decision #003

**About device handling:**
1. **Auto-detection is robust** - Works across Mac, RTX 5090, H200 without changes
2. **Logging is essential** - `log_gpu_memory()` made validation trivial
3. **Progressive testing works** - qwen_tiny caught the bug before expensive qwen_remote run
4. **User intuition matters** - "I do worry that fall back to cpu is a red flag" was correct

### Technical Debt Resolved

**From TODO_REFACTORING.md:**
- ✅ Device-aware operator loading (validated on H200)
- ✅ Device-aware intervention function (working on GPU)
- ✅ GPU memory profiling (32GB tracked correctly)
- ✅ **Config preset device handling** (NEW - not in original TODO)

**From Decision #003:**
- ✅ Test on actual H100/H200 to validate GPU operator loading
- ✅ Operators load to GPU successfully (0.70 GB each)
- ✅ No silent CPU fallback

### Future Implications

**For experiments:**
- Same config preset now works on any hardware (local Mac, cloud GPU)
- No manual device specification needed
- GPU utilization is visible and verifiable

**For collaboration:**
- New users don't need to understand device handling
- Code "just works" on different hardware
- Logging makes debugging trivial

**For paper:**
- Can claim framework is hardware-agnostic
- Validated on 143GB GPU (scales to future models)
- Progressive testing strategy is documented

### Validation Criteria

**How we'll know this was the right decision:**
1. ✅ **qwen_tiny loads to GPU on H200** (validated)
2. ✅ **qwen_remote completes all 3 phases on GPU** (validated)
3. ✅ **Operators load to GPU, not CPU** (validated: 0.70GB each on CUDA)
4. ✅ **Interventions produce sentiment shifts** (validated: clear positive/negative changes)
5. ✅ **Total cost < $2** (actual: ~$0.25 for 10 min @ $1.50/hr)

---

**Last Updated:** 2025-11-05 01:35 UTC
**Decisions Logged:** 4
