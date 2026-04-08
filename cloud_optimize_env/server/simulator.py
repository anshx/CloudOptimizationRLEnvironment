"""
Deterministic rule-based simulator.
Takes the current architecture state + an action, returns the new state.
"""

import copy
from typing import Any, Dict, Optional, Tuple

# Redis tier capacity ratios (relative)
REDIS_TIER_CAPACITY = {"small": 1.0, "medium": 2.5, "large": 5.0}
REDIS_TIER_COST = {"small": 22.0, "medium": 43.0, "large": 86.0}
REDIS_TIERS_ORDERED = ["small", "medium", "large"]

# Staging savings when switching to office hours
STAGING_OFFICE_HOURS_SAVINGS_RATIO = 0.70  # save 70% of cost


def apply_action(
    state: Dict[str, Any],
    action_type: str,
    target: Optional[str],
    value_int: Optional[int],
    value_str: Optional[str],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Apply an action to the architecture state.
    Returns (new_state, error_message).
    error_message is None if action succeeded.
    """
    state = copy.deepcopy(state)

    if action_type == "set_instances":
        return _set_instances(state, target, value_int)
    elif action_type == "set_min_instances":
        return _set_min_instances(state, target, value_int)
    elif action_type == "shrink_redis_tier":
        return _shrink_redis_tier(state, value_str)
    elif action_type == "upgrade_redis_tier":
        return _upgrade_redis_tier(state, value_str)
    elif action_type == "remove_db_replica":
        return _remove_db_replica(state, value_int)
    elif action_type == "add_db_replica":
        return _add_db_replica(state, value_int)
    elif action_type == "set_staging_schedule":
        return _set_staging_schedule(state, value_str)
    elif action_type == "finish":
        return state, None
    else:
        return state, f"Unknown action_type: {action_type}"


def _set_instances(
    state: Dict[str, Any], target: Optional[str], count: Optional[int]
) -> Tuple[Dict[str, Any], Optional[str]]:
    if target not in ("api_service", "worker_service"):
        return state, f"set_instances target must be 'api_service' or 'worker_service', got '{target}'"
    if count is None or count < 1:
        return state, "set_instances requires value_int >= 1"

    svc = state["services"][target]
    old_count = svc["instances"]

    if count == old_count:
        return state, None  # no-op

    if count < svc["min_instances"]:
        return state, f"Cannot set {target} below min_instances ({svc['min_instances']})"

    ratio = old_count / count
    svc["instances"] = count
    svc["avg_utilization_pct"] = round(svc["_original_avg_util"] * (svc["_original_instances"] / count), 2)
    svc["peak_utilization_pct"] = round(svc["_original_peak_util"] * (svc["_original_instances"] / count), 2)

    return state, None


def _set_min_instances(
    state: Dict[str, Any], target: Optional[str], count: Optional[int]
) -> Tuple[Dict[str, Any], Optional[str]]:
    if target not in ("api_service", "worker_service"):
        return state, f"set_min_instances target must be 'api_service' or 'worker_service', got '{target}'"
    if count is None or count < 1:
        return state, "set_min_instances requires value_int >= 1"

    svc = state["services"][target]
    svc["min_instances"] = count

    # If current instances below new min, bump them up
    if svc["instances"] < count:
        svc["instances"] = count
        svc["avg_utilization_pct"] = round(svc["_original_avg_util"] * (svc["_original_instances"] / count), 2)
        svc["peak_utilization_pct"] = round(svc["_original_peak_util"] * (svc["_original_instances"] / count), 2)

    return state, None


def _shrink_redis_tier(
    state: Dict[str, Any], new_tier: Optional[str]
) -> Tuple[Dict[str, Any], Optional[str]]:
    redis = state["datastores"]["redis"]
    current_tier = redis["tier"]

    if new_tier not in REDIS_TIER_CAPACITY:
        return state, f"Invalid redis tier: '{new_tier}'. Valid: {list(REDIS_TIER_CAPACITY.keys())}"

    current_idx = REDIS_TIERS_ORDERED.index(current_tier)
    new_idx = REDIS_TIERS_ORDERED.index(new_tier)

    if new_idx >= current_idx:
        return state, f"Can only shrink Redis. Current: {current_tier}, requested: {new_tier}"

    old_capacity = REDIS_TIER_CAPACITY[current_tier]
    new_capacity = REDIS_TIER_CAPACITY[new_tier]
    ratio = old_capacity / new_capacity

    redis["tier"] = new_tier
    redis["tier_cost"] = REDIS_TIER_COST[new_tier]
    redis["avg_utilization_pct"] = round(redis["_original_avg_util"] * (REDIS_TIER_CAPACITY[redis["_original_tier"]] / new_capacity), 2)
    redis["peak_utilization_pct"] = round(redis["_original_peak_util"] * (REDIS_TIER_CAPACITY[redis["_original_tier"]] / new_capacity), 2)

    return state, None


def _upgrade_redis_tier(
    state: Dict[str, Any], new_tier: Optional[str]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Upgrade Redis to a larger tier (recovery action)."""
    redis = state["datastores"]["redis"]
    current_tier = redis["tier"]

    if new_tier not in REDIS_TIER_CAPACITY:
        return state, f"Invalid redis tier: '{new_tier}'. Valid: {list(REDIS_TIER_CAPACITY.keys())}"

    current_idx = REDIS_TIERS_ORDERED.index(current_tier)
    new_idx = REDIS_TIERS_ORDERED.index(new_tier)

    if new_idx <= current_idx:
        return state, f"Can only upgrade Redis with this action. Current: {current_tier}, requested: {new_tier}. Use shrink_redis_tier to downgrade."

    new_capacity = REDIS_TIER_CAPACITY[new_tier]

    redis["tier"] = new_tier
    redis["tier_cost"] = REDIS_TIER_COST[new_tier]
    redis["avg_utilization_pct"] = round(redis["_original_avg_util"] * (REDIS_TIER_CAPACITY[redis["_original_tier"]] / new_capacity), 2)
    redis["peak_utilization_pct"] = round(redis["_original_peak_util"] * (REDIS_TIER_CAPACITY[redis["_original_tier"]] / new_capacity), 2)

    return state, None


def _remove_db_replica(
    state: Dict[str, Any], count: Optional[int]
) -> Tuple[Dict[str, Any], Optional[str]]:
    pg = state["datastores"]["postgres"]
    remove_count = count if count is not None else 1

    if remove_count < 1:
        return state, "remove_db_replica requires value_int >= 1"

    new_replicas = pg["replicas"] - remove_count
    min_replicas = state["constraints"]["min_db_replicas"]

    if new_replicas < 0:
        return state, f"Cannot remove {remove_count} replicas, only {pg['replicas']} exist"

    # Allow going below min — but it will trigger a hard violation in the grader
    pg["replicas"] = new_replicas

    return state, None


def _add_db_replica(
    state: Dict[str, Any], count: Optional[int]
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Add back a DB replica (recovery action)."""
    pg = state["datastores"]["postgres"]
    add_count = count if count is not None else 1

    if add_count < 1:
        return state, "add_db_replica requires value_int >= 1"

    max_replicas = pg.get("_original_replicas", 2) + 1  # can't add more than original + 1
    new_replicas = pg["replicas"] + add_count

    if new_replicas > max_replicas:
        return state, f"Cannot add {add_count} replicas, max is {max_replicas}"

    pg["replicas"] = new_replicas

    return state, None


def _set_staging_schedule(
    state: Dict[str, Any], schedule: Optional[str]
) -> Tuple[Dict[str, Any], Optional[str]]:
    staging = state["environments"]["staging_env"]

    if schedule not in ("always_on", "office_hours"):
        return state, f"Invalid schedule: '{schedule}'. Valid: 'always_on', 'office_hours'"

    if schedule == staging["schedule"]:
        return state, None  # no-op

    staging["schedule"] = schedule

    if schedule == "office_hours":
        staging["monthly_cost"] = round(staging["_original_cost"] * (1 - STAGING_OFFICE_HOURS_SAVINGS_RATIO), 2)
    else:
        staging["monthly_cost"] = staging["_original_cost"]

    return state, None


# ── Metrics calculation ────────────────────────────────────────────


def calculate_cost(state: Dict[str, Any]) -> Dict[str, float]:
    """Calculate per-component and total monthly cost."""
    breakdown = {}

    for name, svc in state["services"].items():
        breakdown[name] = svc["instances"] * svc["instance_cost"]

    pg = state["datastores"]["postgres"]
    breakdown["postgres"] = pg["instance_cost"]
    breakdown["postgres_replicas"] = pg["replicas"] * pg["replica_cost"]

    redis = state["datastores"]["redis"]
    breakdown["redis"] = redis["tier_cost"]

    staging = state["environments"]["staging_env"]
    breakdown["staging_env"] = staging["monthly_cost"]

    breakdown["total"] = sum(breakdown.values())
    return breakdown


def calculate_latency(state: Dict[str, Any]) -> float:
    """
    Calculate estimated p95 latency in milliseconds.

    Includes cascading effects:
    - Redis over capacity → cache misses → increased DB load → higher DB latency
    - DB over capacity → slow queries → API request queuing → higher API latency
    """
    try:
        base_latency = 15.0  # base network + processing

        # ── Redis cascade: overloaded cache → more DB queries ──
        redis = state["datastores"]["redis"]
        redis_peak = redis["peak_utilization_pct"]

        if redis_peak <= 75:
            cache_miss_extra = 0.0
            db_load_multiplier = 1.0  # normal cache hit rate
        elif redis_peak <= 90:
            cache_miss_extra = 15.0
            db_load_multiplier = 1.3  # 30% more queries hit DB
        elif redis_peak <= 100:
            cache_miss_extra = 40.0
            db_load_multiplier = 1.6  # 60% more queries hit DB
        else:
            cache_miss_extra = 80.0   # severe cache miss storm
            db_load_multiplier = 2.5  # DB gets hammered

        # ── DB contribution (with cascade from Redis) ──
        pg = state["datastores"]["postgres"]
        effective_db_peak = pg["peak_utilization_pct"] * db_load_multiplier

        db_latency = 10.0 + (effective_db_peak / 100.0) * 20.0

        # Replica removal impact
        original_replicas = pg.get("_original_replicas", pg["replicas"])
        removed = original_replicas - pg["replicas"]
        if removed > 0:
            db_latency += removed * 3.0

        # If DB is overloaded, queries queue up massively
        if effective_db_peak > 100:
            db_latency += (effective_db_peak - 100) * 5.0  # +5ms per % over capacity

        # ── DB cascade: slow DB → API requests queue up ──
        api = state["services"]["api_service"]
        api_peak = api["peak_utilization_pct"]

        # Slow DB responses cause API servers to hold connections longer
        db_slowdown_factor = 1.0
        if db_latency > 50:
            db_slowdown_factor = 1.3   # DB is slow, API holds connections
        if db_latency > 100:
            db_slowdown_factor = 1.8   # DB is very slow, API queues build

        effective_api_peak = api_peak * db_slowdown_factor

        if effective_api_peak <= 70:
            api_latency = 5.0
        elif effective_api_peak <= 85:
            api_latency = 15.0
        elif effective_api_peak <= 100:
            api_latency = 60.0
        else:
            api_latency = 60.0 + (effective_api_peak - 100) * 10.0  # degrades linearly

        total = base_latency + api_latency + db_latency + cache_miss_extra
        return round(total, 1)
    except Exception:
        return 999.0


def calculate_availability(state: Dict[str, Any]) -> float:
    """Calculate estimated availability percentage."""
    try:
        base = 0.999  # single component availability

        # API servers: parallel redundancy
        api_count = state["services"]["api_service"]["instances"]
        api_avail = 1.0 - (1.0 - base) ** api_count

        # Workers: parallel redundancy
        worker_count = state["services"]["worker_service"]["instances"]
        worker_avail = 1.0 - (1.0 - base) ** worker_count

        # DB: primary + replicas
        pg = state["datastores"]["postgres"]
        db_components = 1 + pg["replicas"]
        db_avail = 1.0 - (1.0 - base) ** db_components

        # System = serial chain of api * worker * db
        system_avail = api_avail * worker_avail * db_avail
        return round(system_avail * 100, 4)
    except Exception:
        return 99.0


def check_violations(
    state: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Check for hard and soft violations.
    Returns (hard_violations, soft_violations).
    """
    hard = 0
    soft = 0
    constraints = state["constraints"]

    # 1. Check capacity — any service over 100% peak utilization
    for name, svc in state["services"].items():
        if svc["peak_utilization_pct"] > 100:
            hard += 1

    # 2. Check redis capacity
    redis = state["datastores"]["redis"]
    if redis["peak_utilization_pct"] > 100:
        hard += 1

    # 3. Check DB replica minimum
    pg = state["datastores"]["postgres"]
    if pg["replicas"] < constraints["min_db_replicas"]:
        hard += 1

    # 4. Check latency SLO
    latency = calculate_latency(state)
    max_latency = constraints["max_latency_p95_ms"]
    if latency > max_latency * 1.5:
        hard += 1  # way over SLO
    elif latency > max_latency:
        soft += 1  # slightly over SLO

    # 5. Check availability
    availability = calculate_availability(state)
    if availability < constraints["min_availability_pct"]:
        soft += 1

    return hard, soft


# ── Traffic events ─────────────────────────────────────────────────


def get_active_event(
    state: Dict[str, Any], current_step: int
) -> Optional[Dict[str, Any]]:
    """Check if a traffic event is active at the current step."""
    events = state.get("traffic_events", [])
    for event in events:
        start = event["trigger_step"]
        end = start + event.get("duration", 1)
        if start <= current_step < end:
            return event
    return None


def apply_traffic_event(
    state: Dict[str, Any], event: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Temporarily apply a traffic event multiplier to peak utilization.
    Returns a modified COPY of the state for violation checking.
    The original state is NOT modified (event is transient).
    """
    stressed = copy.deepcopy(state)
    multiplier = event["multiplier"]
    affects = event.get("affects", [])

    for target in affects:
        if target in stressed["services"]:
            svc = stressed["services"][target]
            svc["peak_utilization_pct"] = round(
                svc["peak_utilization_pct"] * multiplier, 2
            )
        elif target == "postgres":
            pg = stressed["datastores"]["postgres"]
            pg["peak_utilization_pct"] = round(
                pg["peak_utilization_pct"] * multiplier, 2
            )
        elif target == "redis":
            redis = stressed["datastores"]["redis"]
            redis["peak_utilization_pct"] = round(
                redis["peak_utilization_pct"] * multiplier, 2
            )

    return stressed


def prepare_state(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare internal state from a scenario config.
    Stores original values for utilization calculations.
    """
    state = copy.deepcopy(scenario)

    # Store originals for services
    for name, svc in state["services"].items():
        svc["_original_instances"] = svc["instances"]
        svc["_original_avg_util"] = svc["avg_utilization_pct"]
        svc["_original_peak_util"] = svc["peak_utilization_pct"]

    # Store originals for redis
    redis = state["datastores"]["redis"]
    redis["_original_tier"] = redis["tier"]
    redis["_original_avg_util"] = redis["avg_utilization_pct"]
    redis["_original_peak_util"] = redis["peak_utilization_pct"]

    # Store originals for postgres
    pg = state["datastores"]["postgres"]
    pg["_original_replicas"] = pg["replicas"]

    # Store originals for staging
    staging = state["environments"]["staging_env"]
    staging["_original_cost"] = staging["monthly_cost"]

    return state
