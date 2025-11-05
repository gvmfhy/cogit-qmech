# Experiment Tracker Integration Plan

## Problem Statement

**Current Issue**: Fragile filename-based data versioning causes experimental contamination:
- Phase 1 saves: `quantum_states_{model_id}_latest.json`
- Phase 2 loads: `quantum_states_{model_id}_latest.json`
- **No validation** that quantum_dim, layer, num_prompts match!

**Real Bug Encountered**:
```
# Old Phase 1: quantum_dim=1500
quantum_states_qwen2.5-3B_latest.json  (1500-d data)

# New config: quantum_dim=5333
# Phase 2 silently loaded 1500-d data → trained 1500-d operators
# Phase 3 tried to load 1500-d operators into 5333-d architecture → CRASH
```

## Solution: Experiment Fingerprinting

### Core Concept

Each experiment configuration has a unique **fingerprint** based on:
```python
ExperimentFingerprint(
    model_name="Qwen/Qwen2.5-3B",
    layer=31,
    quantum_dim=5333,
    prompts_per_class=50,
    seed=42
)
# → Hash: "a3f7e291" (8-char SHA256)
```

### File Naming Convention

**Before (fragile)**:
```
quantum_states_qwen2.5-3B_latest.json
encoder_projection_qwen2.5-3B_latest.pt
unitary_pos_to_neg_qwen2.5-3B_latest.pt
```

**After (robust)**:
```
# Phase 1:
quantum_states_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_055525.json
encoder_projection_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_055525.pt

# Phase 2:
unitary_pos_to_neg_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_061430.pt
unitary_neg_to_pos_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_061430.pt

# Metadata (tracks lineage):
data/sentiment_quantum/experiment_metadata.json
```

### Metadata Tracking

**experiment_metadata.json**:
```json
{
  "a3f7e291": {
    "fingerprint": {
      "model_name": "Qwen/Qwen2.5-3B",
      "layer": 31,
      "quantum_dim": 5333,
      "prompts_per_class": 50,
      "seed": 42
    },
    "phase1": {
      "quantum_states_file": "quantum_states_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_055525.json",
      "projection_file": "encoder_projection_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_055525.pt",
      "timestamp": "2025-11-05T05:55:25"
    },
    "phase2": {
      "pos_to_neg_file": "unitary_pos_to_neg_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_061430.pt",
      "neg_to_pos_file": "unitary_neg_to_pos_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_061430.pt",
      "final_fidelity": 0.9456,
      "timestamp": "2025-11-05T06:14:30"
    },
    "phase3": {
      "results_file": "quantum_results_Qwen_Qwen2_5_3B_L31_5333d_50prompts_20251105_062010.json",
      "timestamp": "2025-11-05T06:20:10"
    }
  }
}
```

## Implementation Plan

### Phase 1: Modify `quantum_phase1_collect.py`

**Changes to `QuantumDataCollector.save_quantum_data()`** (line 193-260):

```python
def save_quantum_data(
    self,
    positive_quantum: List[torch.Tensor],
    negative_quantum: List[torch.Tensor],
    stats: dict
) -> Path:
    """Save quantum states with experiment fingerprint"""

    from src.experiment_tracker import ExperimentTracker, ExperimentFingerprint

    # Create output directory
    output_dir = ROOT / self.config.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment fingerprint
    fingerprint = ExperimentFingerprint(
        model_name=self.config.model_name,
        layer=self.config.target_layer,
        quantum_dim=self.config.quantum_dim,
        prompts_per_class=len(positive_quantum),
        seed=self.config.seed
    )

    # Initialize tracker
    tracker = ExperimentTracker(output_dir)

    # Generate filenames using fingerprint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_suffix = fingerprint.to_filename_suffix()

    output_file = output_dir / f"quantum_states_{filename_suffix}_{timestamp}.json"
    projection_file = output_dir / f"encoder_projection_{filename_suffix}_{timestamp}.pt"

    # Prepare data for saving
    data = {
        'positive_quantum_states': [...],
        'negative_quantum_states': [...],
        'config': {...},
        'separation_stats': stats,
        'fingerprint': fingerprint.to_dict(),  # ← NEW: embed fingerprint
        'timestamp': timestamp
    }

    # Save files
    with open(output_file, 'w') as f:
        json.dump(data, f)

    self.encoder.save_projection_matrix(projection_file)

    # Register with tracker
    tracker.register_phase1(
        fingerprint=fingerprint,
        quantum_states_file=output_file,
        projection_file=projection_file
    )

    print(f"✓ Registered experiment {fingerprint.to_hash()}")
    print(f"✓ Saved quantum states to {output_file.name}")

    return output_file
```

### Phase 2: Modify `quantum_phase2_train.py`

**Changes to `UnitaryOperatorTrainer.load_quantum_states()`** (line 58-100):

```python
def load_quantum_states(self):
    """Load quantum states from Phase 1 with fingerprint validation"""

    from src.experiment_tracker import ExperimentTracker, ExperimentFingerprint

    data_dir = ROOT / self.config.data_dir

    # Create fingerprint from current config
    current_fingerprint = ExperimentFingerprint(
        model_name=self.config.model_name,
        layer=self.config.target_layer,
        quantum_dim=self.config.quantum_dim,
        prompts_per_class=self.config.prompts_per_class,
        seed=self.config.seed
    )

    # Initialize tracker
    tracker = ExperimentTracker(data_dir)

    # Try to get Phase 1 files for this fingerprint
    phase1_files = tracker.get_phase1_files(current_fingerprint)

    if phase1_files is None:
        raise FileNotFoundError(
            f"❌ No Phase 1 data found for experiment {current_fingerprint.to_hash()}\n"
            f"   Model: {self.config.model_name} (Layer {self.config.target_layer})\n"
            f"   Quantum dim: {self.config.quantum_dim:,}-d\n"
            f"   Prompts/class: {self.config.prompts_per_class}\n\n"
            f"   Run Phase 1 first:\n"
            f"   python experiments/sentiment/quantum_phase1_collect.py --preset {args.preset}"
        )

    # Load quantum states
    latest_file = phase1_files['quantum_states']

    print(f"\n[Loading Quantum States]")
    print(f"  Experiment: {current_fingerprint.to_hash()}")
    print(f"  File: {latest_file.name}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Validate fingerprint matches (paranoid check)
    if 'fingerprint' in data:
        loaded_fp = ExperimentFingerprint.from_dict(data['fingerprint'])
        if loaded_fp != current_fingerprint:
            raise ValueError(
                f"❌ Fingerprint mismatch!\n"
                f"   Expected: {current_fingerprint.to_hash()}\n"
                f"   Got: {loaded_fp.to_hash()}\n"
                f"   This should never happen - metadata may be corrupted!"
            )

    # Reconstruct complex tensors
    self.positive_states = torch.stack([...])
    self.negative_states = torch.stack([...])

    print(f"✓ Loaded {len(self.positive_states)} positive quantum states")
    print(f"✓ Loaded {len(self.negative_states)} negative quantum states")
    print(f"✓ Quantum dimension: {self.positive_states.shape[1]:,}-d")

    # Store fingerprint for Phase 2 registration
    self.fingerprint = current_fingerprint
```

**Changes to operator saving** (after training completes):

```python
# After training completes (line ~180):
tracker.register_phase2(
    fingerprint=self.fingerprint,
    pos_to_neg_file=pos_neg_save_path,
    neg_to_pos_file=neg_pos_save_path,
    final_fidelity=final_fidelity
)
```

### Phase 3: Modify `quantum_phase3_test.py`

Similar pattern - load using fingerprint from tracker.

### Phase 4: Modify `run_full_pipeline.py`

**Changes to `check_existing_data()`** (line 50-97):

```python
def check_existing_data(self) -> bool:
    """Check if Phase 1 data exists with matching fingerprint"""

    from src.experiment_tracker import ExperimentTracker, ExperimentFingerprint

    data_dir = ROOT / self.config.data_dir

    # Create fingerprint
    fingerprint = ExperimentFingerprint(
        model_name=self.config.model_name,
        layer=self.config.target_layer,
        quantum_dim=self.config.quantum_dim,
        prompts_per_class=self.config.prompts_per_class,
        seed=self.config.seed
    )

    # Initialize tracker
    tracker = ExperimentTracker(data_dir)

    # Check if Phase 1 data exists for this fingerprint
    phase1_files = tracker.get_phase1_files(fingerprint)

    if phase1_files is None:
        print(f"\n[Data Check] No existing Phase 1 data found for experiment {fingerprint.to_hash()}")
        return False

    # Check age
    file_age = time.time() - phase1_files['quantum_states'].stat().st_mtime
    age_minutes = file_age / 60

    print(f"\n[Data Check] Found existing Phase 1 data")
    print(f"  Experiment: {fingerprint.to_hash()}")
    print(f"  File: {phase1_files['quantum_states'].name}")
    print(f"  Age: {age_minutes:.1f} minutes")

    if age_minutes > 60:
        print(f"  ⚠️  Data is older than 1 hour (may be stale)")

    # Auto-yes mode or prompt user
    if self.auto_yes:
        print(f"\n[Auto-yes mode] Reusing existing Phase 1 data")
        return True

    response = input(f"\nReuse existing Phase 1 data? [Y/n]: ")
    return response.lower() != 'n'
```

## Benefits

### 1. **Prevents Data Contamination**
```python
# BEFORE: Silent mismatch
Phase 1: quantum_dim=1500 → quantum_states_qwen2.5-3B_latest.json
Phase 2: quantum_dim=5333 → LOADS WRONG FILE! → Crash

# AFTER: Explicit validation
Phase 1: quantum_dim=1500 → quantum_states_...1500d...json (hash: abc12345)
Phase 2: quantum_dim=5333 → Creates new fingerprint (hash: def67890)
         → tracker.get_phase1_files() returns None
         → FAILS FAST with clear error message
```

### 2. **Experiment Lineage**
Track full pipeline:
```bash
$ python -c "from src.experiment_tracker import *; t = ExperimentTracker('data/sentiment_quantum'); print(t.get_experiment_summary('a3f7e291'))"

Experiment a3f7e291:
  Model: Qwen/Qwen2.5-3B (Layer 31)
  Quantum dim: 5,333-d
  Prompts/class: 50
  ✓ Phase 1: 2025-11-05T05:55:25
  ✓ Phase 2: 2025-11-05T06:14:30 (fidelity: 0.9456)
  ✓ Phase 3: 2025-11-05T06:20:10
```

### 3. **Parallel Safety**
Run multiple experiments simultaneously without collision:
```bash
# Terminal 1: Qwen2.5-3B Layer 31, 5333-d
python run_full_pipeline.py --preset qwen_local

# Terminal 2: Qwen2.5-3B Layer 29, 5333-d
python run_full_pipeline.py --preset qwen_layer29

# Terminal 3: Qwen3-4B Layer 33, 2000-d
python run_full_pipeline.py --preset qwen3_4b

# All tracked separately - no collision!
```

### 4. **Reproducibility**
Exact experiment configuration embedded in filename and metadata.

## Migration Strategy

### Option A: Gradual Migration (Recommended)
1. ✅ Create `src/experiment_tracker.py` (DONE)
2. Add tracker to Phase 1 (keeps backward compatibility)
3. Test Phase 1 → Phase 2 pipeline
4. Add tracker to Phase 2
5. Add tracker to Phase 3
6. Update run_full_pipeline.py

### Option B: Big Bang
- Modify all phases at once
- Requires re-running all existing experiments
- Higher risk

## Testing Plan

1. **Unit tests** for ExperimentFingerprint:
   - Hash stability
   - Equality checks
   - Filename sanitization

2. **Integration tests**:
   - Phase 1 → Phase 2 handoff
   - Dimension mismatch detection
   - Parallel experiment safety

3. **Manual verification**:
   - Run tiny preset end-to-end
   - Verify metadata.json structure
   - Test reusing Phase 1 data

## Backward Compatibility

**Old files** (e.g., `quantum_states_qwen2.5-3B_latest.json`):
- Keep existing files
- Add migration script to register them in metadata
- Phase 2 tries fingerprint lookup first, falls back to old naming

**New behavior**:
- All new experiments use fingerprint-based naming
- Metadata automatically tracks lineage
- Fail-fast on dimension mismatches

## Next Steps

1. Wait for current experiments to finish (Qwen3-4B sweep, Pythia eval, Qwen pipeline)
2. Commit experiment_tracker.py to git
3. Integrate into Phase 1 (test with tiny preset)
4. Integrate into Phase 2 (test Phase 1→2 pipeline)
5. Integrate into Phase 3
6. Update run_full_pipeline.py
7. Document usage in README.md
