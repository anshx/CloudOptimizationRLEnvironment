"""
Deterministic grader for cloud cost optimization.

Final score is continuous between 0.0 and 1.0 based on:
- Cost savings vs target (0.50 weight)
- Violation severity (0.25 weight) — continuous decay, not binary
- SLO compliance (0.15 weight) — continuous based on latency margin
- Efficiency (0.10 weight) — fewer steps = higher score
"""

from typing import Any, Dict, Tuple


def calculate_step_reward(
    baseline_cost: float,
    previous_cost: float,
    new_cost: float,
    new_hard_violations: int,
    new_soft_violations: int,
    is_noop: bool = False,
) -> float:
    """
    Calculate per-step reward in [0.0, 1.0] range.
    Higher = better. 0.5 = neutral (no change). Above 0.5 = saved money safely.
    Below 0.5 = caused violations or wasted a step.
    """
    # Start at 0.5 (neutral baseline)
    reward = 0.5

    # Cost savings bonus: up to +0.4 for large savings in one step
    cost_improvement = max(0.0, previous_cost - new_cost) / baseline_cost if baseline_cost > 0 else 0
    reward += min(0.4, cost_improvement * 3.0)

    # Cost increase penalty (recovery actions cost money)
    cost_increase = max(0.0, new_cost - previous_cost) / baseline_cost if baseline_cost > 0 else 0
    reward -= min(0.2, cost_increase * 2.0)

    # Violation penalties
    reward -= 0.15 * min(new_hard_violations, 3)  # up to -0.45 for 3 violations
    reward -= 0.05 * min(new_soft_violations, 3)   # up to -0.15 for 3 soft violations

    # Churn penalty for no-ops
    if is_noop:
        reward -= 0.05

    return round(max(0.0, min(1.0, reward)), 4)


def calculate_final_score(
    baseline_cost: float,
    final_cost: float,
    cost_target_pct: float,
    total_hard_violations: int,
    total_soft_violations: int,
    steps_taken: int,
    max_steps: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate deterministic final score.
    Always returns a continuous float in [0.0, 1.0].

    Components:
    - Cost savings (0.50): continuous, proportional to savings vs target
    - Violations (0.25): continuous decay — more violations = lower score
    - SLO compliance (0.15): continuous — fewer soft violations = higher
    - Efficiency (0.10): continuous — fewer steps = higher
    """
    # ── Cost savings component (0.50 weight) ──
    # Continuous: 0.0 at no savings, scales linearly, 0.50 at target, can slightly exceed
    savings_pct = (baseline_cost - final_cost) / baseline_cost * 100 if baseline_cost > 0 else 0

    if savings_pct <= 0:
        cost_score = 0.0
    elif savings_pct >= cost_target_pct:
        # Bonus for exceeding target, up to 0.50
        overshoot = (savings_pct - cost_target_pct) / cost_target_pct
        cost_score = min(0.50, 0.45 + 0.05 * min(overshoot, 1.0))
    else:
        # Linear scale from 0.0 to 0.45 as savings approach target
        cost_score = 0.45 * (savings_pct / cost_target_pct)

    # ── Violation component (0.25 weight) ──
    # Continuous decay: 0.25 at 0 violations, decays exponentially
    # 1 violation = 0.12, 2 = 0.06, 3 = 0.03, 5+ ≈ 0.0
    if total_hard_violations == 0:
        violation_score = 0.25
    else:
        violation_score = 0.25 * (0.5 ** total_hard_violations)

    # ── SLO compliance component (0.15 weight) ──
    # Continuous: 0.15 at 0 soft violations, decays with each
    # 1 soft = 0.10, 2 = 0.07, 3 = 0.04, 5+ ≈ 0.01
    if total_soft_violations == 0:
        slo_score = 0.15
    else:
        slo_score = 0.15 * (0.6 ** total_soft_violations)

    # ── Efficiency component (0.10 weight) ──
    # Continuous: fewer steps relative to max = higher score
    if max_steps > 0 and steps_taken > 0:
        efficiency_ratio = 1.0 - (steps_taken / max_steps)
        efficiency_score = 0.10 * max(0.0, efficiency_ratio)
    else:
        efficiency_score = 0.0

    total = cost_score + violation_score + slo_score + efficiency_score
    total = max(0.0, min(1.0, round(total, 4)))

    breakdown = {
        "cost_savings_component": round(cost_score, 4),
        "violation_component": round(violation_score, 4),
        "slo_component": round(slo_score, 4),
        "efficiency_component": round(efficiency_score, 4),
        "total": total,
    }
    return total, breakdown
