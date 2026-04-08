"""
CloudOptimizeEnvironment — the core OpenEnv environment.
Implements reset(), step(), state for cloud cost optimization.
"""

import copy
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server import Environment

from cloud_optimize_env.models import CloudOptimizeObservation

from .grader import calculate_final_score, calculate_step_reward
from .scenarios import get_scenario, list_scenarios
from .simulator import (
    apply_action,
    apply_traffic_event,
    calculate_availability,
    calculate_cost,
    calculate_latency,
    check_violations,
    get_active_event,
    prepare_state,
)


class CloudOptimizeEnvironment(Environment):
    """
    RL environment for cloud infrastructure cost optimization.

    The agent observes an over-provisioned architecture and takes actions
    to reduce cost without violating SLOs or compliance constraints.

    Scored on: cost savings (0.60) + no violations (0.25) + SLO (0.10) + efficiency (0.05).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state_data: Dict[str, Any] = {}
        self._baseline_cost: float = 0.0
        self._previous_cost: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = 10
        self._task_id: str = ""
        self._difficulty: str = ""
        self._description: str = ""
        self._cost_target_pct: float = 0.0
        self._done: bool = False
        self._total_hard_violations: int = 0
        self._total_soft_violations: int = 0
        self._prev_hard_violations: int = 0
        self._prev_soft_violations: int = 0
        self._action_history: list = []
        self._active_event_desc: Optional[str] = None
        self._episode_id: str = str(uuid.uuid4())
        self._env_state = State(episode_id=self._episode_id, step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset environment with a task scenario."""
        task_id = task or kwargs.get("task_id", "obvious_waste")

        scenario = get_scenario(task_id)
        self._state_data = prepare_state(scenario)
        self._task_id = scenario["task_id"]
        self._difficulty = scenario["difficulty"]
        self._description = scenario["description"]
        self._max_steps = scenario["max_steps"]
        self._cost_target_pct = scenario["cost_target_pct"]

        cost_breakdown = calculate_cost(self._state_data)
        self._baseline_cost = cost_breakdown["total"]
        self._previous_cost = self._baseline_cost

        self._step_count = 0
        self._done = False
        self._total_hard_violations = 0
        self._total_soft_violations = 0
        self._prev_hard_violations = 0
        self._prev_soft_violations = 0
        self._active_event_desc = None
        self._action_history = []
        self._episode_id = episode_id or str(uuid.uuid4())
        self._env_state = State(episode_id=self._episode_id, step_count=0)

        return self._build_observation(reward=0.0, error=None)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one optimization action."""
        if self._done:
            return self._build_observation(
                reward=0.0, error="Episode is done. Call reset() to start a new one."
            )

        self._step_count += 1
        self._env_state.step_count = self._step_count

        # Check for active traffic event BEFORE processing action
        active_event = get_active_event(self._state_data, self._step_count)
        if active_event:
            self._active_event_desc = active_event.get("description", "Traffic surge")
        else:
            self._active_event_desc = None

        # Parse action fields
        action_data = action.model_dump() if hasattr(action, "model_dump") else dict(action)
        action_type = action_data.get("action_type", "")
        target = action_data.get("target")
        value_int = action_data.get("value_int")
        value_str = action_data.get("value_str")

        # Handle finish — but still account for active event in final score
        if action_type == "finish":
            return self._finish()

        # Apply action
        new_state, error = apply_action(
            self._state_data, action_type, target, value_int, value_str
        )

        if error:
            # Action failed — return error, small penalty
            # But still check max steps
            if self._step_count >= self._max_steps:
                return self._finish()
            return self._build_observation(reward=-0.02, error=error)

        # Action succeeded — update state
        self._state_data = new_state

        # Record action
        self._action_history.append({
            "step": self._step_count,
            "action_type": action_type,
            "target": target,
            "value_int": value_int,
            "value_str": value_str,
        })

        # Apply traffic event stress for violation checking
        if active_event:
            stressed_state = apply_traffic_event(self._state_data, active_event)
        else:
            stressed_state = self._state_data

        # Calculate new metrics
        cost_breakdown = calculate_cost(self._state_data)  # cost unaffected by transient spike
        new_cost = cost_breakdown["total"]
        hard_v, soft_v = check_violations(stressed_state)  # violations against stressed state

        # Track only NEW violations (not repeated from same state)
        new_hard = max(0, hard_v - self._prev_hard_violations)
        new_soft = max(0, soft_v - self._prev_soft_violations)
        self._total_hard_violations += new_hard
        self._total_soft_violations += new_soft
        self._prev_hard_violations = hard_v
        self._prev_soft_violations = soft_v

        # Calculate step reward
        is_noop = (new_cost == self._previous_cost and new_hard == 0 and new_soft == 0)
        reward = calculate_step_reward(
            baseline_cost=self._baseline_cost,
            previous_cost=self._previous_cost,
            new_cost=new_cost,
            new_hard_violations=new_hard,
            new_soft_violations=new_soft,
            is_noop=is_noop,
        )

        self._previous_cost = new_cost

        # Auto-finish if max steps reached
        if self._step_count >= self._max_steps:
            return self._finish()

        return self._build_observation(reward=reward, error=None)

    def _finish(self) -> Observation:
        """End the episode and compute final score."""
        self._done = True

        # Check if traffic event is active during finish
        active_event = get_active_event(self._state_data, self._step_count)
        if active_event:
            stressed = apply_traffic_event(self._state_data, active_event)
            hard_v, soft_v = check_violations(stressed)
            new_hard = max(0, hard_v - self._prev_hard_violations)
            new_soft = max(0, soft_v - self._prev_soft_violations)
            self._total_hard_violations += new_hard
            self._total_soft_violations += new_soft

        cost_breakdown = calculate_cost(self._state_data)
        final_cost = cost_breakdown["total"]

        score, breakdown = calculate_final_score(
            baseline_cost=self._baseline_cost,
            final_cost=final_cost,
            cost_target_pct=self._cost_target_pct,
            total_hard_violations=self._total_hard_violations,
            total_soft_violations=self._total_soft_violations,
            steps_taken=self._step_count,
            max_steps=self._max_steps,
        )

        return self._build_observation(reward=score, error=None, final_score=score)

    @property
    def state(self) -> State:
        return self._env_state

    def _build_observation(
        self,
        reward: float,
        error: Optional[str],
        final_score: Optional[float] = None,
    ) -> Observation:
        """Build observation from current state."""
        try:
            # If a traffic event is active, show stressed metrics
            active_event = get_active_event(self._state_data, self._step_count)
            if active_event:
                display_state = apply_traffic_event(self._state_data, active_event)
            else:
                display_state = self._state_data

            cost_breakdown = calculate_cost(self._state_data)  # cost is not affected by spike
            current_cost = cost_breakdown["total"]
            latency = calculate_latency(display_state)  # latency shows stressed value
            availability = calculate_availability(display_state)
            hard_v, soft_v = check_violations(display_state)  # violations show stressed state

            savings_pct = (
                (self._baseline_cost - current_cost) / self._baseline_cost * 100
                if self._baseline_cost > 0
                else 0.0
            )

            # Strip internal _original fields from observation
            services = {}
            for name, svc in self._state_data.get("services", {}).items():
                services[name] = {
                    k: v for k, v in svc.items() if not k.startswith("_")
                }

            datastores = {}
            for name, ds in self._state_data.get("datastores", {}).items():
                datastores[name] = {
                    k: v for k, v in ds.items() if not k.startswith("_")
                }

            environments = {}
            for name, env in self._state_data.get("environments", {}).items():
                environments[name] = {
                    k: v for k, v in env.items() if not k.startswith("_")
                }

            return CloudOptimizeObservation(
                done=self._done,
                reward=round(reward, 4),
                task_id=self._task_id,
                difficulty=self._difficulty,
                task_description=self._description,
                services=services,
                datastores=datastores,
                environments=environments,
                baseline_cost=round(self._baseline_cost, 2),
                current_cost=round(current_cost, 2),
                cost_breakdown={
                    k: round(v, 2) for k, v in cost_breakdown.items()
                },
                latency_p95_ms=round(latency, 1),
                availability_pct=round(availability, 4),
                workload=self._state_data.get("workload", {}),
                constraints={
                    k: v
                    for k, v in self._state_data.get("constraints", {}).items()
                    if not k.startswith("_")
                },
                step_number=self._step_count,
                max_steps=self._max_steps,
                remaining_steps=self._max_steps - self._step_count,
                cost_savings_pct=round(savings_pct, 2),
                cost_target_pct=self._cost_target_pct,
                hard_violations=hard_v,
                soft_violations=soft_v,
                last_action_error=error,
                active_event=self._active_event_desc,
            )
        except Exception as e:
            return CloudOptimizeObservation(
                done=self._done,
                reward=0.0,
                last_action_error=f"Observation build error: {str(e)}",
            )
