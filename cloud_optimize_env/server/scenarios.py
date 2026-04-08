"""
Three task scenarios: easy, medium, hard.
Each defines a starting architecture the agent must optimize.
"""

SCENARIOS = {
    "obvious_waste": {
        "task_id": "obvious_waste",
        "difficulty": "easy",
        "description": (
            "Worker fleet is heavily underutilized, Redis is oversized, "
            "staging runs 24/7. Find the obvious savings."
        ),
        "services": {
            "api_service": {
                "instances": 4,
                "min_instances": 2,
                "instance_cost": 124.0,
                "avg_utilization_pct": 35.0,
                "peak_utilization_pct": 55.0,
            },
            "worker_service": {
                "instances": 10,
                "min_instances": 2,
                "instance_cost": 62.0,
                "avg_utilization_pct": 12.0,
                "peak_utilization_pct": 25.0,
            },
        },
        "datastores": {
            "postgres": {
                "instance_tier": "large",
                "instance_cost": 280.0,
                "replicas": 1,
                "replica_cost": 280.0,
                "avg_utilization_pct": 40.0,
                "peak_utilization_pct": 60.0,
            },
            "redis": {
                "tier": "large",
                "tier_cost": 86.0,
                "avg_utilization_pct": 15.0,
                "peak_utilization_pct": 25.0,
            },
        },
        "environments": {
            "staging_env": {
                "schedule": "always_on",
                "monthly_cost": 450.0,
            },
        },
        "workload": {
            "avg_rps": 200,
            "peak_rps": 400,
            "read_write_ratio": 0.8,
            "peak_multiplier": 2.0,
            "pattern": "steady",
        },
        "constraints": {
            "max_latency_p95_ms": 200,
            "min_availability_pct": 99.9,
            "min_db_replicas": 1,
            "compliance": [],
        },
        "cost_target_pct": 40.0,
        "max_steps": 8,
        "traffic_events": [
            # Mild spike at step 5. Most reasonable architectures survive.
            # Teaches the agent that events happen.
            {
                "trigger_step": 5,
                "description": "Marketing campaign drives 1.5x normal traffic",
                "multiplier": 1.5,
                "duration": 2,  # affects steps 5 and 6
                "affects": ["api_service", "worker_service", "postgres", "redis"],
            },
        ],
    },
    "constrained_downsizing": {
        "task_id": "constrained_downsizing",
        "difficulty": "medium",
        "description": (
            "API is somewhat overprovisioned, DB has 2 replicas but one may "
            "be removable. Peak traffic matters. Find safe savings."
        ),
        "services": {
            "api_service": {
                "instances": 6,
                "min_instances": 3,
                "instance_cost": 124.0,
                "avg_utilization_pct": 30.0,
                "peak_utilization_pct": 65.0,
            },
            "worker_service": {
                "instances": 4,
                "min_instances": 2,
                "instance_cost": 62.0,
                "avg_utilization_pct": 45.0,
                "peak_utilization_pct": 70.0,
            },
        },
        "datastores": {
            "postgres": {
                "instance_tier": "large",
                "instance_cost": 280.0,
                "replicas": 2,
                "replica_cost": 280.0,
                "avg_utilization_pct": 50.0,
                "peak_utilization_pct": 75.0,
            },
            "redis": {
                "tier": "medium",
                "tier_cost": 43.0,
                "avg_utilization_pct": 55.0,
                "peak_utilization_pct": 70.0,
            },
        },
        "environments": {
            "staging_env": {
                "schedule": "always_on",
                "monthly_cost": 450.0,
            },
        },
        "workload": {
            "avg_rps": 800,
            "peak_rps": 2000,
            "read_write_ratio": 0.6,
            "peak_multiplier": 2.5,
            "pattern": "spiky",
        },
        "constraints": {
            "max_latency_p95_ms": 150,
            "min_availability_pct": 99.95,
            "min_db_replicas": 1,
            "compliance": [],
        },
        "cost_target_pct": 25.0,
        "max_steps": 10,
        "traffic_events": [
            # Significant spike at step 5. Punishes aggressive cuts made before step 5.
            # If agent removed replica AND shrunk API before this, latency spikes.
            {
                "trigger_step": 5,
                "description": "Flash sale causes 2x traffic surge",
                "multiplier": 2.0,
                "duration": 2,  # affects steps 5 and 6
                "affects": ["api_service", "worker_service", "postgres", "redis"],
            },
        ],
    },
    "bursty_workload_trap": {
        "task_id": "bursty_workload_trap",
        "difficulty": "hard",
        "description": (
            "Average traffic is low but peak is sharp. Removing too much "
            "capacity causes SLO failure. Find savings without falling into "
            "the over-optimization trap."
        ),
        "services": {
            "api_service": {
                "instances": 8,
                "min_instances": 4,
                "instance_cost": 124.0,
                "avg_utilization_pct": 20.0,
                "peak_utilization_pct": 85.0,
            },
            "worker_service": {
                "instances": 6,
                "min_instances": 3,
                "instance_cost": 62.0,
                "avg_utilization_pct": 25.0,
                "peak_utilization_pct": 80.0,
            },
        },
        "datastores": {
            "postgres": {
                "instance_tier": "large",
                "instance_cost": 280.0,
                "replicas": 2,
                "replica_cost": 280.0,
                "avg_utilization_pct": 35.0,
                "peak_utilization_pct": 70.0,
            },
            "redis": {
                "tier": "large",
                "tier_cost": 86.0,
                "avg_utilization_pct": 30.0,
                "peak_utilization_pct": 75.0,
            },
        },
        "environments": {
            "staging_env": {
                "schedule": "always_on",
                "monthly_cost": 450.0,
            },
        },
        "workload": {
            "avg_rps": 500,
            "peak_rps": 3000,
            "read_write_ratio": 0.5,
            "peak_multiplier": 6.0,
            "pattern": "bursty",
        },
        "constraints": {
            "max_latency_p95_ms": 100,
            "min_availability_pct": 99.99,
            "min_db_replicas": 1,
            "compliance": ["pci_dss"],
        },
        "cost_target_pct": 15.0,
        "max_steps": 12,
        "traffic_events": [
            # Major spike at step 4. The trap within the trap.
            # Average utilization LOOKS low, peak is already high,
            # and this spike makes any shrinking deadly.
            {
                "trigger_step": 4,
                "description": "Unexpected viral event causes 3x traffic surge",
                "multiplier": 3.0,
                "duration": 3,  # affects steps 4, 5, and 6
                "affects": ["api_service", "worker_service", "postgres", "redis"],
            },
            # Second smaller spike later, to punish late aggressive moves too
            {
                "trigger_step": 9,
                "description": "End-of-day batch processing surge",
                "multiplier": 1.5,
                "duration": 2,
                "affects": ["worker_service", "postgres"],
            },
        ],
    },
}


def get_scenario(task_id: str) -> dict:
    if task_id not in SCENARIOS:
        raise ValueError(
            f"Unknown task: {task_id}. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[task_id]


def list_scenarios() -> list:
    return list(SCENARIOS.keys())
