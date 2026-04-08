"""
CloudOptimize Environment Client.

Connects to a running CloudOptimize environment server via WebSocket.
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import CloudOptimizeAction, CloudOptimizeObservation, CloudOptimizeState


class CloudOptimizeEnv(EnvClient[CloudOptimizeAction, CloudOptimizeObservation, CloudOptimizeState]):
    """
    Client for the CloudOptimize Environment.

    Example:
        >>> with CloudOptimizeEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task="obvious_waste")
        ...     result = env.step(CloudOptimizeAction(
        ...         action_type="set_instances",
        ...         target="worker_service",
        ...         value_int=3
        ...     ))
        ...     print(result.observation.metadata["current_cost"])
    """

    def _step_payload(self, action: CloudOptimizeAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward", 0.0)

        # Build observation from the serialized fields
        obs = CloudOptimizeObservation(
            done=done,
            reward=reward,
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            task_description=obs_data.get("task_description", ""),
            services=obs_data.get("services", {}),
            datastores=obs_data.get("datastores", {}),
            environments=obs_data.get("environments", {}),
            baseline_cost=obs_data.get("baseline_cost", 0.0),
            current_cost=obs_data.get("current_cost", 0.0),
            cost_breakdown=obs_data.get("cost_breakdown", {}),
            latency_p95_ms=obs_data.get("latency_p95_ms", 0.0),
            availability_pct=obs_data.get("availability_pct", 99.99),
            workload=obs_data.get("workload", {}),
            constraints=obs_data.get("constraints", {}),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            remaining_steps=obs_data.get("remaining_steps", 10),
            cost_savings_pct=obs_data.get("cost_savings_pct", 0.0),
            cost_target_pct=obs_data.get("cost_target_pct", 0.0),
            hard_violations=obs_data.get("hard_violations", 0),
            soft_violations=obs_data.get("soft_violations", 0),
            last_action_error=obs_data.get("last_action_error"),
            active_event=obs_data.get("active_event"),
        )
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> CloudOptimizeState:
        return CloudOptimizeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", ""),
            baseline_cost=payload.get("baseline_cost", 0.0),
            current_cost=payload.get("current_cost", 0.0),
        )
