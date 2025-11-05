#!/usr/bin/env python3
"""
Automated evaluation of quantum steering operators.

Runs baseline vs learned operators across a prompt set, logging sentiment
shift (SiEBERT) and perplexity (model NLL) for quick iteration on a single GPU.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import pipeline

from config import QuantumConfig
from experiments.sentiment.quantum_phase3_test import QuantumInterventionSystem


ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT / "prompts" / "diverse_prompts_50.json"
RESULTS_DIR = ROOT / "results" / "quantum_intervention"


class DiagonalUnitaryOperator(torch.nn.Module):
    """Structured random unitary using diagonal complex phases."""

    def __init__(self, dim: int, device: torch.device, seed: int):
        super().__init__()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        phases = torch.rand(dim, generator=generator, device=device) * 2 * torch.pi
        diag = torch.exp(1j * phases).to(torch.complex64)
        self.register_buffer("diag", diag)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.diag


def load_prompts(num_prompts: int) -> List[str]:
    """Load prompts from canonical prompt file, repeating as needed."""

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            "Prompt file not found. Ensure Phase 1 has generated prompts or "
            "create prompts/diverse_prompts_50.json."
        )

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = data["positive_prompts"]
    negatives = data["negative_prompts"]

    pool = positives + negatives
    if not pool:
        raise ValueError("No prompts available in prompt file.")

    if num_prompts <= len(pool):
        return pool[:num_prompts]

    # Repeat prompts if requested more than available
    repeats = (num_prompts + len(pool) - 1) // len(pool)
    expanded = (pool * repeats)[:num_prompts]
    return expanded


def compute_perplexity(system: QuantumInterventionSystem, prompt: str, completion: str) -> float:
    """Compute perplexity of completion conditioned on prompt using the steering model."""

    model = system.adapter.model
    with torch.no_grad():
        prompt_tokens = model.to_tokens(prompt)
        full_text = prompt + completion
        tokens = model.to_tokens(full_text)

        logits = model(tokens, return_type="logits")
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]

        prompt_len = prompt_tokens.shape[1]
        # Only score completion tokens
        completion_log_probs = log_probs[prompt_len - 1 :]
        completion_targets = targets[prompt_len - 1 :]

        token_log_probs = completion_log_probs.gather(
            1, completion_targets.unsqueeze(-1)
        ).squeeze(-1)

        if token_log_probs.numel() == 0:
            return float("inf")

        nll = -token_log_probs.mean()
        perplexity = torch.exp(nll).item()
        return float(perplexity)


def sentiment_pipeline(device: torch.device):
    """Create cached sentiment classifier."""

    kwargs: Dict[str, int] = {}
    if device.type == "cuda":
        kwargs["device"] = 0

    return pipeline(
        "sentiment-analysis",
        model="siebert/sentiment-roberta-large-english",
        **kwargs,
    )


def evaluate_prompt(
    system: QuantumInterventionSystem,
    sentiment_fn,
    prompt: str,
    blend_ratio: float,
    max_tokens: int,
    random_ops: Tuple[torch.nn.Module, torch.nn.Module],
):
    baseline = system.run_baseline(prompt, max_tokens=max_tokens)
    neg_completion = system.run_with_quantum_intervention(
        prompt,
        system.operator_pos_to_neg,
        blend_ratio=blend_ratio,
        max_tokens=max_tokens,
    )
    pos_completion = system.run_with_quantum_intervention(
        prompt,
        system.operator_neg_to_pos,
        blend_ratio=blend_ratio,
        max_tokens=max_tokens,
    )
    rand_neg_completion = system.run_with_quantum_intervention(
        prompt,
        random_ops[0],
        blend_ratio=blend_ratio,
        max_tokens=max_tokens,
    )
    rand_pos_completion = system.run_with_quantum_intervention(
        prompt,
        random_ops[1],
        blend_ratio=blend_ratio,
        max_tokens=max_tokens,
    )

    results = {}
    for label, completion in (
        ("baseline", baseline),
        ("pos_to_neg", neg_completion),
        ("neg_to_pos", pos_completion),
        ("rand_pos_to_neg", rand_neg_completion),
        ("rand_neg_to_pos", rand_pos_completion),
    ):
        sentiment = sentiment_fn(prompt + " " + completion)[0]
        perplexity = compute_perplexity(system, prompt, completion)
        results[label] = {
            "text": completion,
            "sentiment": sentiment["label"],
            "sentiment_score": float(sentiment["score"]),
            "perplexity": perplexity,
        }

    return results


def aggregate(records: List[Dict[str, Dict]]) -> Dict[str, Dict[str, float]]:
    def aggregate_for(label: str) -> Dict[str, float]:
        sentiments = [r[label]["sentiment"] for r in records]
        scores = [r[label]["sentiment_score"] for r in records]
        perplexities = [r[label]["perplexity"] for r in records]

        positive_rate = np.mean([s == "POSITIVE" for s in sentiments])

        return {
            "positive_rate": float(positive_rate),
            "avg_sentiment_score": float(np.mean(scores) if scores else 0.0),
            "avg_perplexity": float(np.mean(perplexities) if perplexities else float("inf")),
        }

    summary = {
        "baseline": aggregate_for("baseline"),
        "pos_to_neg": aggregate_for("pos_to_neg"),
        "neg_to_pos": aggregate_for("neg_to_pos"),
        "rand_pos_to_neg": aggregate_for("rand_pos_to_neg"),
        "rand_neg_to_pos": aggregate_for("rand_neg_to_pos"),
    }

    def bootstrap_diff(a: np.ndarray, b: np.ndarray, trials: int = 2000, seed: int = 42):
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(trials):
            sample_a = rng.choice(a, size=a.size, replace=True)
            sample_b = rng.choice(b, size=b.size, replace=True)
            diffs.append(sample_a.mean() - sample_b.mean())
        diffs = np.array(diffs)
        lower = np.percentile(diffs, 2.5)
        upper = np.percentile(diffs, 97.5)
        return float(diffs.mean()), [float(lower), float(upper)]

    baseline_pos = np.array([r["baseline"]["sentiment"] == "POSITIVE" for r in records], dtype=np.float32)
    learned_neg = np.array([r["pos_to_neg"]["sentiment"] == "POSITIVE" for r in records], dtype=np.float32)
    learned_pos = np.array([r["neg_to_pos"]["sentiment"] == "POSITIVE" for r in records], dtype=np.float32)
    random_neg = np.array([r["rand_pos_to_neg"]["sentiment"] == "POSITIVE" for r in records], dtype=np.float32)
    random_pos = np.array([r["rand_neg_to_pos"]["sentiment"] == "POSITIVE" for r in records], dtype=np.float32)

    lift_neg, ci_neg = bootstrap_diff(learned_neg, baseline_pos)
    lift_neg_rand, ci_neg_rand = bootstrap_diff(learned_neg, random_neg, seed=43)
    lift_pos, ci_pos = bootstrap_diff(learned_pos, baseline_pos, seed=44)
    lift_pos_rand, ci_pos_rand = bootstrap_diff(learned_pos, random_pos, seed=45)

    summary["pos_to_neg"].update(
        {
            "lift_vs_baseline": lift_neg,
            "lift_ci_vs_baseline": ci_neg,
            "lift_vs_random": lift_neg_rand,
            "lift_ci_vs_random": ci_neg_rand,
        }
    )
    summary["neg_to_pos"].update(
        {
            "lift_vs_baseline": lift_pos,
            "lift_ci_vs_baseline": ci_pos,
            "lift_vs_random": lift_pos_rand,
            "lift_ci_vs_random": ci_pos_rand,
        }
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantum intervention operators.")
    parser.add_argument("--preset", default="qwen_remote", help="QuantumConfig preset to use")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--blend-ratio", type=float, default=0.05, help="Blend ratio for interventions")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max new tokens to generate")
    parser.add_argument("--random-seed", type=int, default=1234, help="Seed for random unitary controls")
    parser.add_argument("--model", type=str, default="", help="Override model name (e.g., pythia-410m, pythia-1.4b, Qwen/Qwen3-8B)")

    args = parser.parse_args()

    config = QuantumConfig.from_preset(args.preset)
    if args.model:
        config.model_name = args.model
    system = QuantumInterventionSystem(config)

    sentiment = sentiment_pipeline(system.device)
    prompts = load_prompts(args.num_prompts)

    dim = system.operator_pos_to_neg.quantum_dim
    random_neg_op = DiagonalUnitaryOperator(dim, system.device, seed=args.random_seed)
    random_pos_op = DiagonalUnitaryOperator(dim, system.device, seed=args.random_seed + 1)

    records = []
    for idx, prompt in enumerate(prompts, start=1):
        result = evaluate_prompt(
            system,
            sentiment,
            prompt,
            blend_ratio=args.blend_ratio,
            max_tokens=args.max_tokens,
            random_ops=(random_neg_op, random_pos_op),
        )
        records.append({"prompt": prompt, **result})
        if idx % 10 == 0:
            print(f"Evaluated {idx}/{len(prompts)} prompts")

    summary = aggregate(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"evaluation_{args.preset}_{int(args.blend_ratio*100):02d}_{timestamp}.json"

    payload = {
        "config": {
            **asdict(config),
            "preset": args.preset,
            "blend_ratio": args.blend_ratio,
            "num_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "timestamp": timestamp,
        },
        "summary": summary,
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()

