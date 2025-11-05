#!/usr/bin/env python3
"""
Experiment tracking system for Cogit-QMech

Ensures each experiment has a unique fingerprint based on:
- Model name + layer
- Quantum dimension
- Number of prompts per class
- Random seed

This prevents accidentally mixing data from different experiments.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class ExperimentFingerprint:
    """Unique identifier for an experiment configuration"""

    model_name: str
    layer: int
    quantum_dim: int
    prompts_per_class: int
    seed: int = 42

    def to_hash(self) -> str:
        """Generate 8-character hash from experiment parameters"""
        params = f"{self.model_name}_{self.layer}_{self.quantum_dim}_{self.prompts_per_class}_{self.seed}"
        return hashlib.sha256(params.encode()).hexdigest()[:8]

    def to_filename_suffix(self) -> str:
        """Human-readable filename suffix"""
        # Sanitize model name (remove slashes, special chars)
        safe_model = self.model_name.replace('/', '_').replace('-', '_')
        return f"{safe_model}_L{self.layer}_{self.quantum_dim}d_{self.prompts_per_class}prompts"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentFingerprint':
        """Reconstruct from dictionary"""
        return cls(**data)

    def __eq__(self, other: 'ExperimentFingerprint') -> bool:
        """Check if two experiments are identical"""
        return self.to_hash() == other.to_hash()


class ExperimentTracker:
    """Track experiment metadata and prevent data contamination"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "experiment_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_phase1(
        self,
        fingerprint: ExperimentFingerprint,
        quantum_states_file: Path,
        projection_file: Path
    ):
        """Register Phase 1 data collection"""
        exp_hash = fingerprint.to_hash()

        self.metadata[exp_hash] = {
            "fingerprint": fingerprint.to_dict(),
            "phase1": {
                "quantum_states_file": str(quantum_states_file.name),
                "projection_file": str(projection_file.name),
                "timestamp": datetime.now().isoformat()
            }
        }

        self._save_metadata()
        return exp_hash

    def register_phase2(
        self,
        fingerprint: ExperimentFingerprint,
        pos_to_neg_file: Path,
        neg_to_pos_file: Path,
        final_fidelity: float
    ):
        """Register Phase 2 operator training"""
        exp_hash = fingerprint.to_hash()

        if exp_hash not in self.metadata:
            raise ValueError(
                f"Cannot register Phase 2 for experiment {exp_hash}: "
                f"Phase 1 data not found. Run Phase 1 first."
            )

        self.metadata[exp_hash]["phase2"] = {
            "pos_to_neg_file": str(pos_to_neg_file.name),
            "neg_to_pos_file": str(neg_to_pos_file.name),
            "final_fidelity": final_fidelity,
            "timestamp": datetime.now().isoformat()
        }

        self._save_metadata()

    def register_phase3(
        self,
        fingerprint: ExperimentFingerprint,
        results_file: Path
    ):
        """Register Phase 3 text generation"""
        exp_hash = fingerprint.to_hash()

        if exp_hash not in self.metadata:
            raise ValueError(
                f"Cannot register Phase 3 for experiment {exp_hash}: "
                f"Phase 1 data not found."
            )

        if "phase2" not in self.metadata[exp_hash]:
            raise ValueError(
                f"Cannot register Phase 3 for experiment {exp_hash}: "
                f"Phase 2 operators not found. Run Phase 2 first."
            )

        self.metadata[exp_hash]["phase3"] = {
            "results_file": str(results_file.name),
            "timestamp": datetime.now().isoformat()
        }

        self._save_metadata()

    def get_phase1_files(
        self,
        fingerprint: ExperimentFingerprint
    ) -> Optional[Dict[str, Path]]:
        """Get Phase 1 files for a given experiment"""
        exp_hash = fingerprint.to_hash()

        if exp_hash not in self.metadata or "phase1" not in self.metadata[exp_hash]:
            return None

        phase1 = self.metadata[exp_hash]["phase1"]
        return {
            "quantum_states": self.data_dir / phase1["quantum_states_file"],
            "projection": self.data_dir / phase1["projection_file"]
        }

    def get_phase2_files(
        self,
        fingerprint: ExperimentFingerprint,
        models_dir: Path
    ) -> Optional[Dict[str, Path]]:
        """Get Phase 2 files for a given experiment"""
        exp_hash = fingerprint.to_hash()

        if exp_hash not in self.metadata or "phase2" not in self.metadata[exp_hash]:
            return None

        phase2 = self.metadata[exp_hash]["phase2"]
        return {
            "pos_to_neg": models_dir / phase2["pos_to_neg_file"],
            "neg_to_pos": models_dir / phase2["neg_to_pos_file"]
        }

    def validate_compatibility(
        self,
        fingerprint: ExperimentFingerprint,
        phase: int
    ) -> bool:
        """
        Validate that an experiment fingerprint matches existing data

        Returns:
            True if compatible, False if mismatch
        """
        exp_hash = fingerprint.to_hash()

        if exp_hash not in self.metadata:
            return True  # No existing data, so no conflict

        # Check if fingerprint matches
        existing_fp = ExperimentFingerprint.from_dict(
            self.metadata[exp_hash]["fingerprint"]
        )

        return fingerprint == existing_fp

    def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """List all registered experiments"""
        return self.metadata

    def get_experiment_summary(self, exp_hash: str) -> Optional[str]:
        """Get human-readable summary of experiment"""
        if exp_hash not in self.metadata:
            return None

        exp = self.metadata[exp_hash]
        fp = ExperimentFingerprint.from_dict(exp["fingerprint"])

        summary = []
        summary.append(f"Experiment {exp_hash}:")
        summary.append(f"  Model: {fp.model_name} (Layer {fp.layer})")
        summary.append(f"  Quantum dim: {fp.quantum_dim:,}-d")
        summary.append(f"  Prompts/class: {fp.prompts_per_class}")

        if "phase1" in exp:
            summary.append(f"  ✓ Phase 1: {exp['phase1']['timestamp']}")

        if "phase2" in exp:
            fidelity = exp['phase2'].get('final_fidelity', 'N/A')
            summary.append(f"  ✓ Phase 2: {exp['phase2']['timestamp']} (fidelity: {fidelity:.4f})")

        if "phase3" in exp:
            summary.append(f"  ✓ Phase 3: {exp['phase3']['timestamp']}")

        return "\n".join(summary)
