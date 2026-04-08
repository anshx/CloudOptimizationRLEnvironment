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
the blast radius of each change. This environment simulates that task with cascading
failures and hidden traffic events, letting agents learn cost optimization strategies
through trial and error.

## Key Features

- **Cascading failures**: Redis overload increases DB load, DB slowdown backs up API servers
- **Hidden traffic spikes**: Unpredictable load surges punish over-optimization
- **Recovery actions**: Agents can undo mistakes (scale back up, upgrade tiers)
- **Delayed consequences**: Short-term savings can become long-term disasters
- **Deterministic grading**: Reproducible scores for benchmarking

## Quick Start

```python
from cloud_optimize_env import CloudOptimizeAction, CloudOptimizeEnv

with CloudOptimizeEnv(base_url="https://anshx97-cloud-optimize-env.hf.space").sync() as env:
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

7 optimization/recovery actions + finish:

### Optimize (reduce cost)

| Action | Description | Example |
|--------|-------------|---------|
| `set_instances` | Change instance count for api_service or worker_service | `{"action_type": "set_instances", "target": "worker_service", "value_int": 3}` |
| `set_min_instances` | Change autoscaling floor | `{"action_type": "set_min_instances", "target": "api_service", "value_int": 2}` |
| `shrink_redis_tier` | Downgrade Redis (large -> medium -> small) | `{"action_type": "shrink_redis_tier", "target": "redis", "value_str": "medium"}` |
| `remove_db_replica` | Remove PostgreSQL read replica(s) | `{"action_type": "remove_db_replica", "target": "postgres", "value_int": 1}` |
| `set_staging_schedule` | Switch staging to office hours only | `{"action_type": "set_staging_schedule", "target": "staging_env", "value_str": "office_hours"}` |

### Recover (fix violations, costs more)

| Action | Description | Example |
|--------|-------------|---------|
| `upgrade_redis_tier` | Upgrade Redis (small -> medium -> large) | `{"action_type": "upgrade_redis_tier", "target": "redis", "value_str": "medium"}` |
| `add_db_replica` | Add back a PostgreSQL read replica | `{"action_type": "add_db_replica", "target": "postgres", "value_int": 1}` |

Agents can also use `set_instances` with a higher count to scale back up.

### Finish

| Action | Description | Example |
|--------|-------------|---------|
| `finish` | Submit optimized architecture, end episode | `{"action_type": "finish"}` |

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task name |
| `difficulty` | str | easy / medium / hard |
| `task_description` | str | Human-readable task description |
| `services` | dict | api_service and worker_service state (instances, utilization, cost) |
| `datastores` | dict | postgres (tier, replicas, utilization) and redis (tier, utilization) |
| `environments` | dict | staging_env (schedule, cost) |
| `baseline_cost` | float | Original monthly cost before any changes |
| `current_cost` | float | Current monthly cost |
| `cost_savings_pct` | float | Percentage saved so far |
| `cost_target_pct` | float | Target savings percentage for this task |
| `latency_p95_ms` | float | Estimated p95 latency (includes cascading effects) |
| `availability_pct` | float | Estimated availability percentage |
| `hard_violations` | int | Critical constraint violations |
| `soft_violations` | int | SLO threshold violations |
| `active_event` | str or null | Description of active traffic event (hidden spike in progress) |
| `step_number` | int | Current step in episode |
| `max_steps` | int | Maximum steps allowed |
| `last_action_error` | str or null | Error message if last action was invalid |

## Tasks

### Task 1: Obvious Waste (Easy)

Worker fleet at 12% utilization, Redis oversized, staging running 24/7. Clear savings
opportunities with low risk.

- Baseline cost: $2,212/mo
- Target: save 40%
- Max steps: 8
- Traffic event: 1.5x marketing campaign spike at step 5
- Trap: over-shrinking Redis causes cascading failure (cache miss -> DB overload -> API latency)

### Task 2: Constrained Downsizing (Medium)

API somewhat overprovisioned, DB has 2 replicas (one may be removable), spiky traffic.
Some changes are safe, some are risky.

- Baseline cost: $2,325/mo
- Target: save 25%
- Max steps: 10
- Traffic event: 2x flash sale spike at step 5
- Trap: aggressive cuts before step 5 get punished by the traffic spike

### Task 3: Bursty Workload Trap (Hard)

Average utilization is low (looks wasteful!) but peak traffic is sharp.
Agent must resist over-optimizing based on average numbers.

- Baseline cost: $2,740/mo
- Target: save 15%
- Max steps: 12
- Traffic events: 3x viral spike at step 4, 1.5x batch processing at step 9
- Trap: everything looks shrinkable by average utilization, but peak is already near capacity

## Reward Function

### Per-Step Reward

```
reward = 0.7 * (cost_saved / baseline_cost) - 0.5 * hard_violations - 0.2 * soft_violations
```

Positive reward for cost savings, negative for violations. Agent gets feedback every step.

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

The environment uses deterministic rule-based simulation with cascading effects:

### Compute Resize
New utilization = old utilization * (old count / new count). Peak > 100% = hard failure.

### Redis Tier Change
Capacity scales by tier ratio (small=1x, medium=2.5x, large=5x). Peak memory > 100% triggers
cache miss storm.

### Cascading Failures
- **Redis overload -> DB cascade**: Cache misses increase DB query load by up to 2.5x
- **DB overload -> API cascade**: Slow DB queries cause API servers to hold connections, effectively increasing API utilization by up to 1.8x
- A single bad Redis shrink can take down the entire system

### Traffic Events
Hidden load spikes that fire at specific steps. The agent doesn't know when they'll hit.
During a spike, peak utilization is temporarily multiplied (1.5x to 3x), causing violations
if the agent removed too much capacity.

### DB Replica Removal
Below min_replicas = hard violation. Fewer replicas = slightly higher latency.

### Staging Schedule
Office hours saves 70% of staging cost. Zero production impact. Always safe.

## Setup

### Local Development

```bash
pip install openenv-core
git clone https://github.com/anshx/CloudOptimizationRLEnvironment.git
cd CloudOptimizationRLEnvironment
PYTHONPATH=. uvicorn cloud_optimize_env.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t cloud-optimize-env:latest .
docker run -p 8000:8000 cloud-optimize-env:latest
```

### Verify

```bash
curl http://localhost:8000/health
# {"status":"healthy"}
```

## Running the Baseline

```bash
export HF_TOKEN=<your_huggingface_token>
python inference.py
```

The inference script connects to the deployed HF Space by default. Set `ENV_BASE_URL`
to override:

```bash
ENV_BASE_URL=http://localhost:8000 python inference.py
```

## Baseline Scores

| Task | Model | Score | Steps | Notes |
|------|-------|-------|-------|-------|
| obvious_waste | Qwen 2.5 72B | 0.52 | 8 | Over-optimized Redis, triggered cascade. Partially recovered. |
| constrained_downsizing | Qwen 2.5 72B | 0.71 | 7 | Good first moves, then Redis cascade + traffic spike. |
| bursty_workload_trap | Qwen 2.5 72B | 0.40 | 1 | Too cautious: immediately finished without optimizing. |

The model falls into cascade traps on easy/medium tasks and is paralyzed on the hard
task. This shows room for RL improvement: an agent trained over many episodes would
learn which optimizations are safe and which trigger cascading failures.

## What Makes This Environment Interesting for RL

1. **Delayed consequences**: Removing a replica saves money now, but a traffic spike 3 steps later causes DB saturation
2. **Cascading failures**: One bad Redis shrink takes down the entire stack
3. **Recovery mechanics**: Agents can undo mistakes (scale up, upgrade tiers) but waste steps doing so
4. **Difficulty inversion**: The easy task is harder for greedy agents than the hard task
5. **Exploration vs exploitation**: Agent must balance saving money vs preserving headroom for unknown spikes

## Extensibility

The environment supports custom scenarios via `reset(config={...})`:

```python
result = env.reset(config={
    "services": { ... your architecture ... },
    "workload": { ... your traffic pattern ... },
    "constraints": { ... your SLOs ... },
    "traffic_events": [ ... your hidden spikes ... ],
})
```

Future extensions: more resource types (CDN, WAF, queue), partial observability,
trace-driven replay from real telemetry, multi-agent optimization.
