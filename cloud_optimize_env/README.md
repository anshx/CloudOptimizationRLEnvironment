---
title: CloudOptimizeEnv
emoji: "\u2601\uFE0F"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# CloudOptimizeEnv: Cloud Cost Optimization RL Environment

An OpenEnv-compatible reinforcement learning environment where LLM agents optimize
over-provisioned cloud architectures by resizing, removing, and scheduling components
to reduce monthly cost without violating SLOs.

## Motivation

Every engineering team inherits over-provisioned cloud infrastructure. Reducing cost
without breaking SLOs requires understanding utilization patterns, traffic spikes, and
the blast radius of each change. This environment simulates that task, letting agents
learn cost optimization strategies through trial and error.

## Quick Start

```python
from cloud_optimize_env import CloudOptimizeAction, CloudOptimizeEnv

with CloudOptimizeEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task="obvious_waste")
    obs = result.observation
    print(f"Baseline cost: ${obs.baseline_cost}/mo")

    # Shrink underutilized workers
    result = env.step(CloudOptimizeAction(
        action_type="set_instances",
        target="worker_service",
        value_int=3
    ))
    print(f"New cost: ${result.observation.current_cost}/mo")
    print(f"Reward: {result.reward}")

    # Submit final architecture
    result = env.step(CloudOptimizeAction(action_type="finish"))
    print(f"Final score: {result.reward}")
```

## Action Space

5 optimization actions + finish:

| Action | Description | Example |
|--------|-------------|---------|
| `set_instances` | Change instance count for api_service or worker_service | `{"action_type": "set_instances", "target": "worker_service", "value_int": 3}` |
| `set_min_instances` | Change autoscaling floor | `{"action_type": "set_min_instances", "target": "api_service", "value_int": 2}` |
| `shrink_redis_tier` | Downgrade Redis (large -> medium -> small) | `{"action_type": "shrink_redis_tier", "target": "redis", "value_str": "medium"}` |
| `remove_db_replica` | Remove PostgreSQL read replica(s) | `{"action_type": "remove_db_replica", "target": "postgres", "value_int": 1}` |
| `set_staging_schedule` | Switch staging to office hours only | `{"action_type": "set_staging_schedule", "target": "staging_env", "value_str": "office_hours"}` |
| `finish` | Submit optimized architecture | `{"action_type": "finish"}` |

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task name |
| `difficulty` | str | easy / medium / hard |
| `services` | dict | api_service and worker_service state (instances, utilization, cost) |
| `datastores` | dict | postgres (tier, replicas, utilization) and redis (tier, utilization) |
| `environments` | dict | staging_env (schedule, cost) |
| `baseline_cost` | float | Original monthly cost before any changes |
| `current_cost` | float | Current monthly cost |
| `cost_savings_pct` | float | Percentage saved so far |
| `cost_target_pct` | float | Target savings percentage for this task |
| `latency_p95_ms` | float | Estimated p95 latency |
| `availability_pct` | float | Estimated availability percentage |
| `hard_violations` | int | Critical constraint violations (capacity overflow, below min replicas) |
| `soft_violations` | int | SLO threshold violations (latency slightly over limit) |
| `step_number` | int | Current step in episode |
| `max_steps` | int | Maximum steps allowed |

## Tasks

### Task 1: Obvious Waste (Easy)

Worker fleet at 12% utilization, Redis oversized, staging running 24/7. Clear savings
opportunities with low risk.

- Baseline cost: $2,212/mo
- Target: save 40%
- Max steps: 8
- Expected solution: shrink workers, shrink Redis, schedule staging

### Task 2: Constrained Downsizing (Medium)

API somewhat overprovisioned, DB has 2 replicas (one may be removable), spiky traffic.
Some changes are safe, some are risky.

- Baseline cost: $2,325/mo
- Target: save 25%
- Max steps: 10
- Trap: shrinking Redis at 55% utilization causes cache miss storms under peak load

### Task 3: Bursty Workload Trap (Hard)

Average utilization is low (looks wasteful!) but peak traffic is sharp (6x spike).
Agent must resist over-optimizing based on average numbers.

- Baseline cost: $2,740/mo
- Target: save 15%
- Max steps: 12
- Trap: workers at 25% avg look wasteful, but 80% peak means shrinking causes SLO failure

## Reward Function

### Per-Step Reward

```
reward = 0.7 * (cost_saved / baseline_cost) - 0.5 * hard_violations - 0.2 * soft_violations
```

Positive reward for cost savings, negative for violations. Agent sees score change
after every action.

### Final Score (Deterministic Grader)

```
score = cost_component (0.60) + violation_component (0.25) + slo_component (0.10) + efficiency (0.05)
```

- Cost component: proportional to savings vs target (0.60 max)
- Violation component: 0.25 if zero hard violations, 0.0 otherwise
- SLO component: 0.10 if zero soft violations
- Efficiency: bonus for fewer steps

Scores are always in [0.0, 1.0] range.

## Simulation Rules

The environment uses deterministic rule-based simulation:

- **Compute resize**: new utilization = old utilization * (old count / new count). Peak > 100% = hard failure.
- **Redis shrink**: capacity scales by tier ratio. Peak memory > 100% = cache miss storm.
- **DB replica removal**: below min_replicas = hard violation. Fewer replicas = slight latency increase.
- **Staging schedule**: office hours saves 70% of staging cost. Zero production impact.
- **Latency**: peak utilization thresholds (<=70% safe, 70-85% small penalty, 85-100% major penalty, >100% failure).

## Setup

### Local Development

```bash
pip install openenv-core
git clone <this-repo>
cd cloud_optimize_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t cloud-optimize-env:latest -f server/Dockerfile .
docker run -p 8000:8000 cloud-optimize-env:latest
```

### Verify

```bash
curl http://localhost:8000/health
# {"status":"healthy"}
```

## Baseline Scores

| Task | Model | Score | Steps | Notes |
|------|-------|-------|-------|-------|
| obvious_waste | Qwen 2.5 72B | 0.70 | 8 | Found major savings but also triggered violations (Redis too small, removed only replica) |
| constrained_downsizing | Qwen 2.5 72B | 0.98 | 3 | Clean execution: staging + remove 1 replica + finish |
| bursty_workload_trap | Qwen 2.5 72B | 0.99 | 3 | Correctly avoided the bursty trap, took only safe optimizations |

The easy task is harder for the model than the hard task. The model over-optimizes
obvious waste (greedy), but correctly recognizes danger in bursty workloads (cautious).
This asymmetry shows room for RL improvement in calibrating aggression.

## Extensibility

The environment supports custom scenarios via `reset(config={...})`:

```python
result = env.reset(config={
    "services": { ... your architecture ... },
    "workload": { ... your traffic pattern ... },
    "constraints": { ... your SLOs ... },
})
```

Future extensions: more resource types (CDN, WAF, queue), delayed consequences,
tool-use interface, trace-driven replay from real telemetry.
