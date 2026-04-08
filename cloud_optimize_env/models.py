"""
CloudOptimizeEnv — Pydantic models for Action, Observation, and State.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ── Supporting models ──────────────────────────────────────────────


class ServiceInfo(BaseModel):
    instances: int
    min_instances: int
    instance_cost: float
    avg_utilization_pct: float
    peak_utilization_pct: float


class DatastoreInfo(BaseModel):
    instance_tier: str = "medium"
    instance_cost: float = 0.0
    replicas: int = 0
    replica_cost: float = 0.0
    avg_utilization_pct: float = 0.0
    peak_utilization_pct: float = 0.0


class RedisInfo(BaseModel):
    tier: Literal["small", "medium", "large"]
    tier_cost: float
    avg_utilization_pct: float
    peak_utilization_pct: float


class StagingInfo(BaseModel):
    schedule: Literal["always_on", "office_hours"]
    monthly_cost: float


class WorkloadInfo(BaseModel):
    avg_rps: int
    peak_rps: int
    read_write_ratio: float
    peak_multiplier: float
    pattern: Literal["steady", "spiky", "bursty"]


class ConstraintInfo(BaseModel):
    max_latency_p95_ms: int
    min_availability_pct: float
    min_db_replicas: int
    compliance: List[str] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    api_service: float = 0.0
    worker_service: float = 0.0
    postgres: float = 0.0
    postgres_replicas: float = 0.0
    redis: float = 0.0
    staging_env: float = 0.0
    total: float = 0.0


class ScoreBreakdown(BaseModel):
    cost_savings_component: float = 0.0
    violation_component: float = 0.0
    slo_component: float = 0.0
    efficiency_component: float = 0.0
    total: float = 0.0


# ── Action ─────────────────────────────────────────────────────────


class CloudOptimizeAction(Action):
    """Agent's action to optimize cloud infrastructure."""

    action_type: Literal[
        "set_instances",
        "set_min_instances",
        "shrink_redis_tier",
        "upgrade_redis_tier",
        "remove_db_replica",
        "add_db_replica",
        "set_staging_schedule",
        "finish",
    ]
    target: Optional[str] = None
    value_int: Optional[int] = None
    value_str: Optional[str] = None


# ── Observation ────────────────────────────────────────────────────


class CloudOptimizeObservation(Observation):
    """What the agent sees after each step."""

    # Task info
    task_id: str = ""
    difficulty: str = ""
    task_description: str = ""

    # Architecture state
    services: Dict[str, Any] = Field(default_factory=dict)
    datastores: Dict[str, Any] = Field(default_factory=dict)
    environments: Dict[str, Any] = Field(default_factory=dict)

    # Cost
    baseline_cost: float = 0.0
    current_cost: float = 0.0
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    latency_p95_ms: float = 0.0
    availability_pct: float = 99.99

    # Workload
    workload: Dict[str, Any] = Field(default_factory=dict)

    # Constraints
    constraints: Dict[str, Any] = Field(default_factory=dict)

    # Episode info
    step_number: int = 0
    max_steps: int = 10
    remaining_steps: int = 10
    last_action_error: Optional[str] = None
    active_event: Optional[str] = None  # description of active traffic event, if any

    # Score info
    cost_savings_pct: float = 0.0
    cost_target_pct: float = 0.0
    hard_violations: int = 0
    soft_violations: int = 0


# ── State ──────────────────────────────────────────────────────────


class CloudOptimizeState(State):
    """Internal environment state."""

    task_id: str = ""
    difficulty: str = ""
    baseline_cost: float = 0.0
    current_cost: float = 0.0
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    final_score: Optional[float] = None
