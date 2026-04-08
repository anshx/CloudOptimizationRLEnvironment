"""
Deterministic grader for cloud cost optimization.
Scores based on: cost savings (0.60) + no violations (0.25) + SLO clean (0.10) + efficiency (0.05).
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
    Calculate per-step reward.
    Positive for cost savings, negative for violations.
    """
    # Cost savings reward
    cost_improvement = max(0.0, previous_cost - new_cost) / baseline_cost
    cost_reward = 0.7 * cost_improvement

    # Violation penalties
    hard_penalty = 0.5 * new_hard_violations
    soft_penalty = 0.2 * new_soft_violations

    # Churn penalty for no-ops or repeated actions
    churn_penalty = 0.02 if is_noop else 0.0

    reward = cost_reward - hard_penalty - soft_penalty - churn_penalty
    reward = max(0.0, min(1.0, reward))
    return round(reward, 4)


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
    Calculate deterministic final score (0.0 to 1.0).
    Returns (score, breakdown_dict).
    """
    # Cost savings component (0.60 weight)
    savings_pct = (baseline_cost - final_cost) / baseline_cost * 100 if baseline_cost > 0 else 0
    if savings_pct >= cost_target_pct:
        cost_score = 0.60
    elif savings_pct > 0:
        cost_score = 0.60 * (savings_pct / cost_target_pct)
    else:
        cost_score = 0.0

    # Hard violation component (0.25 weight)
    if total_hard_violations == 0:
        violation_score = 0.25
    else:
        violation_score = 0.0

    # Soft SLO component (0.10 weight)
    if total_soft_violations == 0:
        slo_score = 0.10
    elif total_soft_violations <= 2:
        slo_score = 0.05
    else:
        slo_score = 0.0

    # Efficiency component (0.05 weight)
    if max_steps > 0 and steps_taken > 0:
        efficiency_ratio = 1.0 - (steps_taken / max_steps)
        efficiency_score = 0.05 * max(0.0, efficiency_ratio)
    else:
        efficiency_score = 0.0

    total = cost_score + violation_score + slo_score + efficiency_score
    total = max(0.0, min(1.0, total))

    breakdown = {
        "cost_savings_component": round(cost_score, 4),
        "violation_component": round(violation_score, 4),
        "slo_component": round(slo_score, 4),
        "efficiency_component": round(efficiency_score, 4),
        "total": round(total, 4),
    }
    return total, breakdown
