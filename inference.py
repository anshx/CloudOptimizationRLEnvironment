"""
Inference Script — CloudOptimizeEnv Baseline Agent
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if you are using from_docker_image()

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from cloud_optimize_env import CloudOptimizeAction, CloudOptimizeEnv

# ── Configuration ──────────────────────────────────────────────────

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "cloud_optimize_env"
TASKS = ["obvious_waste", "constrained_downsizing", "bursty_workload_trap"]
MAX_RETRIES = 2  # retries per step if LLM returns invalid JSON
TEMPERATURE = 0.3
MAX_TOKENS = 300

SYSTEM_PROMPT = textwrap.dedent("""
    You are a cloud infrastructure cost optimization agent.

    GOAL: Reduce monthly cloud cost without breaking the system.
    You will receive a reward after each action. Positive = good. Negative = you broke something.
    Maximize your total reward.

    OBSERVATION: Each step shows you the current architecture (services, databases, cache,
    staging), utilization percentages, cost, latency, availability, and any constraint
    violations. Use this information to decide your next action.

    ACTIONS (respond with valid JSON only, no text):

    {"action_type": "set_instances", "target": "api_service"|"worker_service", "value_int": <count>}
    {"action_type": "set_min_instances", "target": "api_service"|"worker_service", "value_int": <count>}
    {"action_type": "shrink_redis_tier", "target": "redis", "value_str": "small"|"medium"}
    {"action_type": "upgrade_redis_tier", "target": "redis", "value_str": "medium"|"large"}
    {"action_type": "remove_db_replica", "target": "postgres", "value_int": <count>}
    {"action_type": "add_db_replica", "target": "postgres", "value_int": <count>}
    {"action_type": "set_staging_schedule", "target": "staging_env", "value_str": "office_hours"|"always_on"}
    {"action_type": "finish"}

    NOTES:
    - If an action returns an error, do not repeat it.
    - Traffic spikes can happen at any time. If you see an active_event, your architecture
      is under stress.
    - "finish" ends the episode and gives your final score.

    Respond with ONLY a single JSON object.
""").strip()


# ── Logging ────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string (remove newlines)
    action_clean = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ───────────────────────────────────────────────


def format_observation(obs) -> str:
    """Format observation into a prompt for the LLM."""
    services = obs.services or {}
    datastores = obs.datastores or {}
    environments = obs.environments or {}
    constraints = obs.constraints or {}

    lines = [
        f"Task: {obs.task_id} ({obs.difficulty})",
        f"Description: {obs.task_description}",
        f"",
        f"CURRENT ARCHITECTURE:",
    ]

    for name, svc in services.items():
        lines.append(
            f"  {name}: {svc['instances']} instances, "
            f"avg_util={svc['avg_utilization_pct']}%, "
            f"peak_util={svc['peak_utilization_pct']}%, "
            f"cost=${svc['instances'] * svc['instance_cost']:.0f}/mo"
        )

    pg = datastores.get("postgres", {})
    lines.append(
        f"  postgres: tier={pg.get('instance_tier','?')}, "
        f"replicas={pg.get('replicas', 0)}, "
        f"avg_util={pg.get('avg_utilization_pct', 0)}%, "
        f"peak_util={pg.get('peak_utilization_pct', 0)}%"
    )

    redis = datastores.get("redis", {})
    lines.append(
        f"  redis: tier={redis.get('tier','?')}, "
        f"avg_util={redis.get('avg_utilization_pct', 0)}%, "
        f"peak_util={redis.get('peak_utilization_pct', 0)}%"
    )

    staging = environments.get("staging_env", {})
    lines.append(
        f"  staging_env: schedule={staging.get('schedule','?')}, "
        f"cost=${staging.get('monthly_cost', 0):.0f}/mo"
    )

    lines.extend([
        f"",
        f"METRICS:",
        f"  Baseline cost: ${obs.baseline_cost:.0f}/mo",
        f"  Current cost: ${obs.current_cost:.0f}/mo",
        f"  Savings so far: {obs.cost_savings_pct:.1f}%",
        f"  Cost target: save {getattr(obs, 'cost_target_pct', constraints.get('cost_target_pct', 0)):.0f}%",
        f"  Latency p95: {obs.latency_p95_ms:.0f}ms (max: {constraints.get('max_latency_p95_ms', '?')}ms)",
        f"  Availability: {obs.availability_pct:.2f}%",
        f"  Hard violations: {obs.hard_violations}",
        f"  Soft violations: {obs.soft_violations}",
        f"",
        f"CONSTRAINTS:",
        f"  Min DB replicas: {constraints.get('min_db_replicas', '?')}",
        f"  Compliance: {constraints.get('compliance', [])}",
        f"",
        f"Step {obs.step_number} of {obs.max_steps}. Choose your next action.",
    ])

    if getattr(obs, 'active_event', None):
        lines.insert(0, f"WARNING: TRAFFIC EVENT ACTIVE: {obs.active_event}\nYour architecture is under stress. Consider recovery actions if violations appeared.\n")

    if obs.last_action_error:
        lines.insert(0, f"ERROR from last action: {obs.last_action_error}\n")

    return "\n".join(lines)


def get_llm_action(
    client: OpenAI, observation_text: str, history: List[str]
) -> Dict[str, Any]:
    """Ask the LLM for the next action. Returns parsed JSON dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent history for context
    for h in history[-4:]:
        messages.append({"role": "assistant", "content": h})

    messages.append({"role": "user", "content": observation_text})

    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()

            # Strip markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            action = json.loads(text)
            return action
        except (json.JSONDecodeError, Exception) as e:
            if attempt == MAX_RETRIES:
                # Fallback: finish
                return {"action_type": "finish"}
            continue


# ── Main loop ─────────────────────────────────────────────────────


async def run_task(
    client: OpenAI, env: CloudOptimizeEnv, task_id: str
) -> None:
    """Run one task episode."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_id)
        obs = result.observation

        max_steps = obs.max_steps or 10

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Format observation for LLM
            obs_text = format_observation(obs)

            # Get LLM decision
            action_dict = get_llm_action(client, obs_text, history)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # Execute action
            action = CloudOptimizeAction(**action_dict)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            # Include error feedback and reward in history so LLM learns from mistakes
            if error:
                history.append(f"ACTION: {action_str} -> ERROR: {error}")
            else:
                history.append(f"ACTION: {action_str} -> reward={reward:.2f}, savings={obs.cost_savings_pct:.1f}%")

            if done:
                break

        # If episode didn't finish naturally, send explicit finish
        if not result.done:
            result = await env.step(CloudOptimizeAction(action_type="finish"))
            obs = result.observation
            rewards.append(result.reward or 0.0)
            steps_taken += 1
            log_step(step=steps_taken, action='{"action_type":"finish"}',
                     reward=result.reward or 0.0, done=result.done, error=obs.last_action_error)

        # Final score = last reward (the grader score from finish)
        score = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await CloudOptimizeEnv.from_docker_image(IMAGE_NAME)
    else:
        # Default: connect to local or HF Space
        base_url = os.getenv("ENV_BASE_URL", "https://anshx97-cloud-optimize-env.hf.space")
        env = CloudOptimizeEnv(base_url=base_url)
        await env.connect()

    try:
        for task_id in TASKS:
            await run_task(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
