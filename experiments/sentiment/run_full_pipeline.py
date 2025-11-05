#!/usr/bin/env python3
"""
Run Full Cogit-QMech Pipeline
Executes Phases 1→2→3→4 with validation and logging

This script provides fail-fast progressive testing:
- Validates each phase before proceeding
- Checks for existing data and prompts for reuse
- Tracks timing and cost estimates
- Auto-appends to EXPERIMENT_LOG.md

Usage:
    python experiments/sentiment/run_full_pipeline.py --preset tiny
    python experiments/sentiment/run_full_pipeline.py --preset qwen_remote
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import argparse
import subprocess
import json

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QuantumConfig


class PipelineRunner:
    """Orchestrate full experimental pipeline with validation"""

    def __init__(self, preset: str, auto_yes: bool = False):
        self.preset = preset
        self.config = QuantumConfig.from_preset(preset)
        self.start_time = time.time()
        self.phase_times = {}
        self.results = {}
        self.auto_yes = auto_yes

        print("\n" + "=" * 70)
        print(f"COGIT-QMECH FULL PIPELINE: {preset.upper()}")
        print("=" * 70)
        self.config.print_summary()

    def check_existing_data(self) -> bool:
        """Check if Phase 1 data exists and is fresh"""
        data_dir = ROOT / self.config.data_dir
        model_id = self.config.model_identifier

        quantum_states_file = data_dir / f"quantum_states_{model_id}_latest.json"
        projection_file = data_dir / f"encoder_projection_{model_id}_latest.pt"

        if not quantum_states_file.exists() or not projection_file.exists():
            print(f"\n[Data Check] No existing Phase 1 data found")
            return False

        # Check quantum dimension compatibility
        try:
            with open(quantum_states_file, 'r') as f:
                data = json.load(f)
                if 'positive_quantum_states' in data and len(data['positive_quantum_states']) > 0:
                    # Get dimension from first state (real + imag components)
                    existing_dim = len(data['positive_quantum_states'][0]['real'])
                    config_dim = self.config.quantum_dim

                    if existing_dim != config_dim:
                        print(f"\n❌ [Dimension Mismatch] Existing data has {existing_dim}-d, config specifies {config_dim}-d")
                        print(f"  Cannot reuse existing data - dimension mismatch would cause operator loading failures")
                        print(f"  Phase 1 will be re-run with {config_dim}-d quantum states")
                        return False
        except Exception as e:
            print(f"\n⚠️  [Warning] Could not verify dimension compatibility: {e}")
            print(f"  Proceeding with caution...")

        # Check age
        file_age = time.time() - quantum_states_file.stat().st_mtime
        age_minutes = file_age / 60

        print(f"\n[Data Check] Found existing Phase 1 data")
        print(f"  File: {quantum_states_file.name}")
        print(f"  Age: {age_minutes:.1f} minutes")

        if age_minutes > 60:
            print(f"  ⚠️  Data is older than 1 hour (may be stale)")

        # Auto-yes mode or prompt user
        if self.auto_yes:
            print(f"\n[Auto-yes mode] Reusing existing Phase 1 data")
            return True

        response = input(f"\nReuse existing Phase 1 data? [Y/n]: ")
        return response.lower() != 'n'

    def run_phase(self, phase_num: int, script_name: str, description: str) -> bool:
        """
        Run a single phase with validation

        Returns:
            True if phase succeeded, False otherwise
        """
        print(f"\n{'=' * 70}")
        print(f"PHASE {phase_num}: {description}")
        print(f"{'=' * 70}\n")

        phase_start = time.time()

        # Build command
        script_path = ROOT / "experiments" / "sentiment" / script_name
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            str(script_path),
            "--preset", self.preset
        ]

        # Set unbuffered env
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        # Run phase
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=False  # Stream output to terminal
            )
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Phase {phase_num} FAILED with exit code {e.returncode}")
            return False

        phase_time = time.time() - phase_start
        self.phase_times[f"Phase{phase_num}"] = phase_time

        print(f"\n✅ Phase {phase_num} completed in {phase_time/60:.1f} minutes")

        return True

    def validate_phase1(self) -> bool:
        """Validate Phase 1 outputs"""
        data_dir = ROOT / self.config.data_dir
        model_id = self.config.model_identifier

        quantum_states_file = data_dir / f"quantum_states_{model_id}_latest.json"
        projection_file = data_dir / f"encoder_projection_{model_id}_latest.pt"

        if not quantum_states_file.exists():
            print(f"❌ Validation failed: {quantum_states_file} not found")
            return False

        if not projection_file.exists():
            print(f"❌ Validation failed: {projection_file} not found")
            return False

        print(f"✓ Phase 1 validation passed")
        return True

    def validate_phase2(self) -> bool:
        """Validate Phase 2 outputs and check fidelity"""
        models_dir = ROOT / self.config.models_dir
        model_id = self.config.model_identifier

        pos_neg_file = models_dir / f"unitary_pos_to_neg_{model_id}_latest.pt"
        neg_pos_file = models_dir / f"unitary_neg_to_pos_{model_id}_latest.pt"

        if not pos_neg_file.exists():
            print(f"❌ Validation failed: {pos_neg_file} not found")
            return False

        if not neg_pos_file.exists():
            print(f"❌ Validation failed: {neg_pos_file} not found")
            return False

        # Check fidelity
        import torch
        checkpoint = torch.load(pos_neg_file, map_location='cpu', weights_only=False)

        if 'training_history' in checkpoint:
            fidelity = checkpoint['training_history']['fidelity_history'][-1]
            print(f"  Final fidelity: {fidelity:.4f}")

            if fidelity < 0.85:
                print(f"❌ Validation failed: Fidelity {fidelity:.4f} < 0.85 threshold")
                return False

        print(f"✓ Phase 2 validation passed")
        return True

    def validate_phase3(self) -> bool:
        """Validate Phase 3 outputs"""
        results_dir = ROOT / self.config.results_dir
        results_file = results_dir / "quantum_results_latest.json"

        if not results_file.exists():
            print(f"❌ Validation failed: {results_file} not found")
            return False

        # Check results are valid JSON
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)

            if 'results' not in data or len(data['results']) == 0:
                print(f"❌ Validation failed: No results in {results_file}")
                return False

            print(f"  Generated {len(data['results'])} test results")
        except json.JSONDecodeError:
            print(f"❌ Validation failed: Invalid JSON in {results_file}")
            return False

        print(f"✓ Phase 3 validation passed")
        return True

    def validate_phase4(self) -> bool:
        """Validate Phase 4 outputs"""
        results_dir = ROOT / self.config.results_dir
        reversibility_file = results_dir / "reversibility_test_latest.json"

        if not reversibility_file.exists():
            print(f"❌ Validation failed: {reversibility_file} not found")
            return False

        print(f"✓ Phase 4 validation passed")
        return True

    def run_pipeline(self) -> bool:
        """Run full pipeline Phases 1→2→3→4"""

        # Phase 1: Data Collection
        skip_phase1 = self.check_existing_data()

        if not skip_phase1:
            if not self.run_phase(1, "quantum_phase1_collect.py", "Data Collection"):
                return False
            if not self.validate_phase1():
                return False
        else:
            print(f"\n⏭️  Skipping Phase 1 (reusing existing data)")
            self.phase_times["Phase1"] = 0

        # Phase 2: Train Operators
        if not self.run_phase(2, "quantum_phase2_train.py", "Train Unitary Operators"):
            return False
        if not self.validate_phase2():
            return False

        # Phase 3: Generate Text Samples (qualitative inspection)
        if not self.run_phase(3, "quantum_phase3_test.py", "Generate Text Samples"):
            return False
        if not self.validate_phase3():
            return False

        # Phase 4: Evaluate Steering Effectiveness (quantitative metrics)
        if not self.run_phase(4, "evaluate_quantum_intervention.py", "Evaluate Steering Effectiveness"):
            return False
        if not self.validate_phase4():
            return False

        return True

    def print_summary(self):
        """Print pipeline summary with timing and costs"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        print(f"\nPreset: {self.preset}")
        print(f"Model: {self.config.model_name}")
        print(f"Quantum dimension: {self.config.quantum_dim:,}")

        print(f"\nTiming:")
        for phase, duration in self.phase_times.items():
            print(f"  {phase}: {duration/60:.1f} min")
        print(f"  Total: {total_time/60:.1f} min")

        # Cost estimate (rough)
        if "qwen_remote" in self.preset:
            gpu_rate = 0.89  # RTX 5090 rate
            cost = (total_time / 3600) * gpu_rate
            print(f"\nEstimated cost: ${cost:.2f} @ ${gpu_rate}/hr")

        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run full Cogit-QMech pipeline")
    parser.add_argument(
        '--preset',
        type=str,
        required=True,
        choices=['tiny', 'local', 'remote', 'qwen_local', 'qwen_tiny', 'qwen_test_layers', 'qwen_remote', 'pythia_410m', 'pythia_test_layers', 'qwen3_4b', 'qwen3_4b_test_layers'],
        help='Configuration preset'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Automatically reuse existing Phase 1 data without prompting'
    )

    args = parser.parse_args()

    runner = PipelineRunner(args.preset, auto_yes=args.yes)

    success = runner.run_pipeline()

    if success:
        runner.print_summary()
        print("\n🎉 Pipeline completed successfully!")
        print(f"\nNext steps:")
        print(f"  - Review results in results/quantum_intervention/")
        print(f"  - Compare to baselines")
        print(f"  - Document in EXPERIMENT_LOG.md")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
