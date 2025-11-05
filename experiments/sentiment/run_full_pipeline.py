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

    def __init__(self, preset: str, auto_yes: bool = False, *,
                 prompts: str = "", neutral_prompts: str = "",
                 blend_ratio: float = None, blend_ratios: list | None = None, max_tokens: int = None,
                 decode_method: str = None, num_prompts: int = None,
                 model: str = None, quantum_dim: int = None,
                 quantum_ratio: float = None):
        self.preset = preset
        self.config = QuantumConfig.from_preset(preset)
        self.start_time = time.time()
        self.phase_times = {}
        self.results = {}
        self.auto_yes = auto_yes
        self.prompts = prompts
        self.neutral_prompts = neutral_prompts
        self.blend_ratio = blend_ratio
        self.blend_ratios = blend_ratios or None
        self.max_tokens = max_tokens
        self.decode_method = decode_method
        self.num_prompts = num_prompts
        self.model = model
        self.quantum_dim = quantum_dim
        self.quantum_ratio = quantum_ratio

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

        # Forward common options based on phase
        if script_name == "quantum_phase1_collect.py":
            if self.prompts:
                cmd += ["--prompts", self.prompts]
            if self.num_prompts:
                cmd += ["--num-prompts", str(self.num_prompts)]
            if self.model:
                cmd += ["--model", self.model]
            if self.quantum_dim:
                cmd += ["--quantum-dim", str(self.quantum_dim)]
            if self.quantum_ratio:
                cmd += ["--quantum-ratio", str(self.quantum_ratio)]
        elif script_name == "quantum_phase3_test.py":
            if self.prompts:
                cmd += ["--prompts", self.prompts]
            if self.num_prompts:
                cmd += ["--num-prompts", str(self.num_prompts)]
            if self.max_tokens:
                cmd += ["--max-tokens", str(self.max_tokens)]
            # Optionally pass blend ratios as single value if provided
            if self.blend_ratios is not None and len(self.blend_ratios) > 0:
                ratios_str = ",".join(str(x) for x in self.blend_ratios)
                cmd += ["--blend-ratios", ratios_str]
            elif self.blend_ratio is not None:
                cmd += ["--blend-ratios", str(self.blend_ratio)]
            if self.model:
                cmd += ["--model", self.model]
        elif script_name == "evaluate_quantum_intervention.py":
            if self.prompts:
                cmd += ["--prompts", self.prompts]
            if self.neutral_prompts:
                cmd += ["--neutral-prompts", self.neutral_prompts]
            if self.num_prompts:
                cmd += ["--num-prompts", str(self.num_prompts)]
            if self.max_tokens:
                cmd += ["--max-tokens", str(self.max_tokens)]
            if self.decode_method:
                cmd += ["--decode-method", self.decode_method]
            if self.blend_ratio is not None:
                cmd += ["--blend-ratio", str(self.blend_ratio)]
            if self.model:
                cmd += ["--model", self.model]

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
        """Validate Phase 4 outputs (evaluation results present)"""
        eval_dir = ROOT / "results" / "quantum_intervention"
        if not eval_dir.exists():
            print(f"❌ Validation failed: {eval_dir} not found")
            return False
        json_files = sorted(eval_dir.glob("evaluation_*.json"))
        if not json_files:
            print(f"❌ Validation failed: no evaluation_*.json found in {eval_dir}")
            return False
        print(f"✓ Phase 4 validation passed ({json_files[-1].name})")
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
    parser.add_argument('--prompts', type=str, default='', help='Path to main prompts JSON (pos+neg)')
    parser.add_argument('--neutral-prompts', type=str, default='', help='Optional path to neutral prompts JSON')
    parser.add_argument('--blend-ratio', type=float, default=None, help='Blend ratio for interventions/evaluation')
    parser.add_argument('--blend-ratios', type=str, default='', help='Comma-separated list for Phase 3 sweep')
    parser.add_argument('--max-tokens', type=int, default=None, help='Max new tokens for generation')
    parser.add_argument('--decode-method', type=str, default=None, choices=[
        'real_component', 'real_imag_avg', 'absolute', 'magnitude'
    ], help='Decoding method for evaluator')
    parser.add_argument('--num-prompts', type=int, default=None, help='Num prompts to use (where applicable)')
    parser.add_argument('--model', type=str, default=None, help='Override model name')
    parser.add_argument('--quantum-dim', type=int, default=None, help='Override quantum dimension for Phase 1')
    parser.add_argument('--quantum-ratio', type=float, default=None, help='Set quantum_dim = round(input_dim * ratio) in Phase 1')
    parser.add_argument('--study-config', type=str, default='', help='JSON file describing a study configuration')

    args = parser.parse_args()

    # Optionally load study config and overlay CLI
    study = {}
    if args.study_config:
        try:
            with open(args.study_config, 'r') as f:
                study = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not read study config {args.study_config}: {e}")
            study = {}

    def pick(key, cli_val, default=None):
        return cli_val if cli_val not in (None, '', []) else study.get(key, default)

    blend_ratios = None
    if args.blend_ratios:
        try:
            blend_ratios = [float(x.strip()) for x in args.blend_ratios.split(',') if x.strip()]
        except ValueError:
            blend_ratios = None
    else:
        br_list = study.get('blend_ratios')
        if isinstance(br_list, list):
            try:
                blend_ratios = [float(x) for x in br_list]
            except Exception:
                blend_ratios = None

    runner = PipelineRunner(
        pick('preset', args.preset),
        auto_yes=args.yes,
        prompts=pick('prompts', args.prompts, ''),
        neutral_prompts=pick('neutral_prompts', args.neutral_prompts, ''),
        blend_ratio=pick('blend_ratio', args.blend_ratio),
        blend_ratios=blend_ratios,
        max_tokens=pick('max_tokens', args.max_tokens),
        decode_method=pick('decode_method', args.decode_method),
        num_prompts=pick('num_prompts', args.num_prompts),
        model=pick('model', args.model),
        quantum_dim=pick('quantum_dim', args.quantum_dim),
        quantum_ratio=pick('quantum_ratio', args.quantum_ratio),
    )

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
