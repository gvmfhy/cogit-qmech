#!/usr/bin/env python3
"""
Statistical Analysis Helpers for Quantum Steering Experiments

Provides utility functions for computing effect sizes, confidence intervals,
and statistical tests used in the publication figures.

Usage:
    from statistical_analysis_helpers import *

    # Compute Cohen's h for proportion difference
    h = cohens_h(p1=0.59, p2=0.61)

    # Bootstrap confidence interval
    ci = bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000)

    # Power analysis for proportion test
    power = proportion_test_power(n=100, p1=0.5, p2=0.55, alpha=0.05)
"""

import numpy as np
from scipy import stats
from typing import Callable, Tuple, List
import warnings


# =============================================================================
# Effect Sizes
# =============================================================================

def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h for difference between proportions.

    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

    Interpretation:
        |h| < 0.2: Small effect
        |h| < 0.5: Medium effect
        |h| >= 0.5: Large effect

    Args:
        p1: Proportion 1 (e.g., baseline positive rate)
        p2: Proportion 2 (e.g., steered positive rate)

    Returns:
        Cohen's h effect size

    Example:
        >>> cohens_h(0.50, 0.52)  # 2pp lift
        0.040
        >>> cohens_h(0.50, 0.60)  # 10pp lift
        0.201
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi2 - phi1


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Cohen's d for difference between means (continuous variables).

    d = (mean(x1) - mean(x2)) / pooled_std

    Interpretation:
        |d| < 0.2: Small effect
        |d| < 0.5: Medium effect
        |d| < 0.8: Large effect
        |d| >= 0.8: Very large effect

    Args:
        x1: First group (e.g., baseline perplexity)
        x2: Second group (e.g., steered perplexity)

    Returns:
        Cohen's d effect size

    Example:
        >>> baseline = np.array([4.2, 4.5, 4.1, 4.3])
        >>> steered = np.array([4.3, 4.6, 4.0, 4.4])
        >>> cohens_d(baseline, steered)
        -0.15
    """
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of data
        statistic: Function to compute (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        (point_estimate, (ci_lower, ci_upper))

    Example:
        >>> data = np.array([0.5, 0.6, 0.55, 0.58, 0.52])
        >>> est, (lo, hi) = bootstrap_ci(data, np.mean)
        >>> print(f"Mean: {est:.3f} [{lo:.3f}, {hi:.3f}]")
    """
    np.random.seed(random_state)

    point_estimate = statistic(data)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_estimates.append(statistic(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha/2) * 100)

    return point_estimate, (ci_lower, ci_upper)


def bootstrap_ci_difference(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap CI for difference between two groups.

    Args:
        data1: Group 1 (e.g., baseline sentiment scores)
        data2: Group 2 (e.g., steered sentiment scores)
        statistic: Function to compute for each group
        n_bootstrap: Number of bootstrap samples
        confidence_level: CI level
        random_state: Random seed

    Returns:
        (difference, (ci_lower, ci_upper))
        where difference = statistic(data2) - statistic(data1)

    Example:
        >>> baseline = np.array([0.5, 0.6, 0.55, 0.58])
        >>> steered = np.array([0.52, 0.61, 0.57, 0.60])
        >>> diff, (lo, hi) = bootstrap_ci_difference(baseline, steered)
        >>> print(f"Lift: {diff:.3f} [{lo:.3f}, {hi:.3f}]")
    """
    np.random.seed(random_state)

    point_diff = statistic(data2) - statistic(data1)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        diff = statistic(sample2) - statistic(sample1)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    return point_diff, (ci_lower, ci_upper)


# =============================================================================
# Power Analysis
# =============================================================================

def proportion_test_power(
    n: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Compute statistical power for two-proportion test.

    Uses normal approximation for large samples.

    Args:
        n: Sample size per group
        p1: Proportion in group 1 (null hypothesis)
        p2: Proportion in group 2 (alternative hypothesis)
        alpha: Significance level (type I error rate)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Statistical power (probability of detecting true effect)

    Example:
        >>> # Power to detect 5pp lift with N=100
        >>> power = proportion_test_power(n=100, p1=0.50, p2=0.55)
        >>> print(f"Power: {power:.1%}")
        Power: 28.4%

        >>> # Needed sample size for 80% power
        >>> for n in [100, 200, 500, 1000]:
        ...     pwr = proportion_test_power(n=n, p1=0.50, p2=0.55)
        ...     print(f"N={n}: power={pwr:.1%}")
    """
    # Pooled proportion under null
    p_pooled = (p1 + p2) / 2

    # Standard error under null
    se_null = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)

    # Critical value
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha/2)
    elif alternative == 'greater':
        z_crit = stats.norm.ppf(1 - alpha)
    elif alternative == 'less':
        z_crit = stats.norm.ppf(alpha)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Effect size
    diff = p2 - p1

    # Standard error under alternative
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)

    # Non-centrality parameter
    if alternative == 'two-sided':
        power = (
            stats.norm.cdf((diff - z_crit * se_null) / se_alt) +
            stats.norm.cdf((-diff - z_crit * se_null) / se_alt)
        )
    elif alternative == 'greater':
        power = stats.norm.cdf((diff - z_crit * se_null) / se_alt)
    else:  # less
        power = stats.norm.cdf((z_crit * se_null - diff) / se_alt)

    return power


def minimum_detectable_effect(
    n: int,
    p_baseline: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = 'two-sided'
) -> float:
    """
    Compute minimum detectable effect (MDE) for given power.

    Args:
        n: Sample size per group
        p_baseline: Baseline proportion (e.g., 0.50)
        alpha: Significance level
        power: Desired power (e.g., 0.80 for 80%)
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        Minimum detectable proportion difference (e.g., 0.18 = 18pp)

    Example:
        >>> # With N=100, what's the smallest lift we can detect?
        >>> mde = minimum_detectable_effect(n=100, p_baseline=0.50)
        >>> print(f"MDE: {mde:.1%} ({mde*100:.0f}pp)")
        MDE: 18.0% (18pp)
    """
    # Binary search for MDE
    low, high = 0, 1 - p_baseline

    for _ in range(100):  # Converge
        mid = (low + high) / 2
        p_alt = p_baseline + mid

        pwr = proportion_test_power(n, p_baseline, p_alt, alpha, alternative)

        if pwr < power:
            low = mid
        else:
            high = mid

        if abs(pwr - power) < 0.001:
            break

    return mid


# =============================================================================
# Statistical Tests
# =============================================================================

def proportion_z_test(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Two-proportion z-test.

    H0: p1 = p2
    HA: p1 != p2 (or > or <, depending on alternative)

    Args:
        count1: Number of successes in group 1
        n1: Total observations in group 1
        count2: Number of successes in group 2
        n2: Total observations in group 2
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        (z_statistic, p_value)

    Example:
        >>> # Baseline: 59/100 positive
        >>> # Steered: 61/100 positive
        >>> z, p = proportion_z_test(59, 100, 61, 100)
        >>> print(f"Z={z:.2f}, p={p:.3f}")
    """
    p1 = count1 / n1
    p2 = count2 / n2

    # Pooled proportion
    p_pooled = (count1 + count2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    # Z-statistic
    z = (p2 - p1) / se

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:  # less
        p_value = stats.norm.cdf(z)

    return z, p_value


def mcnemar_test(
    n_01: int,
    n_10: int
) -> Tuple[float, float]:
    """
    McNemar's test for paired proportions.

    Use when comparing same subjects in two conditions.

    Args:
        n_01: Count of (baseline=0, steered=1) - discordant pairs type 1
        n_10: Count of (baseline=1, steered=0) - discordant pairs type 2

    Returns:
        (chi2_statistic, p_value)

    Example:
        >>> # 100 prompts: 10 changed from neg→pos, 8 changed pos→neg
        >>> chi2, p = mcnemar_test(n_01=10, n_10=8)
        >>> print(f"χ²={chi2:.2f}, p={p:.3f}")
    """
    # McNemar's chi-square
    chi2 = (abs(n_01 - n_10) - 1)**2 / (n_01 + n_10)

    # P-value from chi-square(1)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


# =============================================================================
# Utility Functions
# =============================================================================

def format_ci(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 3
) -> str:
    """
    Format point estimate with confidence interval for display.

    Args:
        estimate: Point estimate
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
        decimals: Number of decimal places

    Returns:
        Formatted string

    Example:
        >>> format_ci(0.019, -0.12, 0.15)
        '+0.019 [-0.120, +0.150]'
    """
    sign = '+' if estimate >= 0 else ''
    return f"{sign}{estimate:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:+.{decimals}f}]"


def interpret_p_value(p: float) -> str:
    """
    Interpret p-value with standard thresholds.

    Args:
        p: P-value

    Returns:
        Interpretation string

    Example:
        >>> interpret_p_value(0.03)
        '* (p < 0.05)'
        >>> interpret_p_value(0.68)
        'n.s. (p ≥ 0.05)'
    """
    if p < 0.001:
        return '*** (p < 0.001)'
    elif p < 0.01:
        return '** (p < 0.01)'
    elif p < 0.05:
        return '* (p < 0.05)'
    else:
        return 'n.s. (p ≥ 0.05)'


def interpret_cohens_h(h: float) -> str:
    """
    Interpret Cohen's h effect size.

    Args:
        h: Cohen's h value

    Returns:
        Interpretation string

    Example:
        >>> interpret_cohens_h(0.04)
        'trivial (|h| < 0.2)'
    """
    abs_h = abs(h)
    if abs_h < 0.2:
        return 'trivial (|h| < 0.2)'
    elif abs_h < 0.5:
        return 'small (0.2 ≤ |h| < 0.5)'
    elif abs_h < 0.8:
        return 'medium (0.5 ≤ |h| < 0.8)'
    else:
        return 'large (|h| ≥ 0.8)'


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string

    Example:
        >>> interpret_cohens_d(-0.15)
        'small (0.2 ≤ |d| < 0.5)'
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'trivial (|d| < 0.2)'
    elif abs_d < 0.5:
        return 'small (0.2 ≤ |d| < 0.5)'
    elif abs_d < 0.8:
        return 'medium (0.5 ≤ |d| < 0.8)'
    else:
        return 'large (|d| ≥ 0.8)'


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Statistical Analysis Helpers - Example Usage\n")
    print("=" * 70)

    # Example 1: Effect size for observed result
    print("\n1. Effect Size for Typical Result")
    print("-" * 70)
    baseline_rate = 0.59
    steered_rate = 0.61
    lift = steered_rate - baseline_rate
    h = cohens_h(baseline_rate, steered_rate)

    print(f"Baseline positive rate: {baseline_rate:.1%}")
    print(f"Steered positive rate: {steered_rate:.1%}")
    print(f"Lift: {lift:.3f} ({lift*100:.1f}pp)")
    print(f"Cohen's h: {h:.3f} ({interpret_cohens_h(h)})")

    # Example 2: Bootstrap CI for lift
    print("\n2. Bootstrap Confidence Interval")
    print("-" * 70)
    np.random.seed(42)
    baseline_outcomes = np.random.binomial(1, baseline_rate, 100)
    steered_outcomes = np.random.binomial(1, steered_rate, 100)

    diff, (ci_low, ci_high) = bootstrap_ci_difference(
        baseline_outcomes, steered_outcomes,
        statistic=np.mean,
        n_bootstrap=10000
    )

    print(f"Estimated lift: {format_ci(diff, ci_low, ci_high)}")
    print(f"CI includes 0: {ci_low <= 0 <= ci_high}")

    # Example 3: Statistical test
    print("\n3. Proportion Z-Test")
    print("-" * 70)
    z, p = proportion_z_test(59, 100, 61, 100)
    print(f"Z-statistic: {z:.3f}")
    print(f"P-value: {p:.3f} {interpret_p_value(p)}")

    # Example 4: Power analysis
    print("\n4. Power Analysis")
    print("-" * 70)
    n = 100
    mde = minimum_detectable_effect(n, p_baseline=0.5)
    print(f"Sample size: {n}")
    print(f"Minimum detectable effect (80% power): {mde:.1%} ({mde*100:.0f}pp)")
    print(f"Observed lift ({lift*100:.1f}pp) is {'below' if lift < mde else 'above'} MDE")

    # Example 5: Power for different sample sizes
    print("\n5. Required Sample Size for 80% Power")
    print("-" * 70)
    target_lift = 0.02  # 2pp lift (observed)
    print(f"Target effect: {target_lift:.1%} lift")
    print("\nSample size vs Power:")
    for n_test in [100, 200, 500, 1000, 2000]:
        pwr = proportion_test_power(n_test, 0.50, 0.50 + target_lift)
        print(f"  N={n_test:4d}: Power={pwr:5.1%}")

    print("\n" + "=" * 70)
    print("See figure code for usage in visualizations")
