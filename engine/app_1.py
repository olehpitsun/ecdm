import csv
import json
import os
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, request


app = Flask(__name__)

APP_URL = os.getenv(
    "APP_URL",
    "http://app-service:8000",
)

PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus:9090",
)

DATA_FILE = Path(
    os.getenv(
        "DATA_FILE",
        "/data/experiment_results.csv",
    )
)

DATA_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RECOVERY_HISTORY_FILE = Path(
    os.getenv(
        "RECOVERY_HISTORY_FILE",
        "/data/recovery_history.json",
    )
)
RECOVERY_HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RECOVERY_PRIOR_ALPHA = float(
    os.getenv("RECOVERY_PRIOR_ALPHA", "1.0")
)
RECOVERY_PRIOR_BETA = float(
    os.getenv("RECOVERY_PRIOR_BETA", "1.0")
)

MIN_AUTONOMOUS_CONFIDENCE = float(
    os.getenv("MIN_AUTONOMOUS_CONFIDENCE", "0.55")
)
MIN_CONFIDENCE_MARGIN = float(
    os.getenv("MIN_CONFIDENCE_MARGIN", "0.03")
)
RECOVERY_VERIFICATION_MAX_ATTEMPTS = int(
    os.getenv("RECOVERY_VERIFICATION_MAX_ATTEMPTS", "20")
)
RECOVERY_VERIFICATION_INTERVAL_SECONDS = float(
    os.getenv("RECOVERY_VERIFICATION_INTERVAL_SECONDS", "0.5")
)
RECOVERY_REQUIRED_CONSECUTIVE_CHECKS = int(
    os.getenv("RECOVERY_REQUIRED_CONSECUTIVE_CHECKS", "3")
)

ACTION_EXECUTION_MODE = os.getenv(
    "ACTION_EXECUTION_MODE",
    "prototype_fault_reset",
)

AUTONOMOUS_ACTION_ALLOWLIST = {
    action.strip()
    for action in os.getenv(
        "AUTONOMOUS_ACTION_ALLOWLIST",
        "Scale Out,Restart Service,Restart Dependency,Circuit Breaker,Rollback,Do Nothing",
    ).split(",")
    if action.strip()
}

_raw_hdcm_weights = {
    "ce": float(os.getenv("HDCM_ALPHA", "0.25")),
    "ch": float(os.getenv("HDCM_BETA", "0.25")),
    "cc": float(os.getenv("HDCM_GAMMA", "0.25")),
    "cr": float(os.getenv("HDCM_DELTA", "0.25")),
}
_weight_sum = sum(max(value, 0.0) for value in _raw_hdcm_weights.values())
if _weight_sum <= 0:
    _raw_hdcm_weights = {"ce": 0.25, "ch": 0.25, "cc": 0.25, "cr": 0.25}
    _weight_sum = 1.0
HDCM_WEIGHTS = {
    key: max(value, 0.0) / _weight_sum
    for key, value in _raw_hdcm_weights.items()
}
HDCM_AGGREGATION_MODE = os.getenv(
    "HDCM_AGGREGATION_MODE",
    "multiplicative",
).strip().lower()

_history_lock = threading.Lock()


# ============================================================
# General utility functions
# ============================================================

def clamp(value):
    return max(
        0.0,
        min(1.0, float(value)),
    )


def _load_recovery_history_unlocked():
    if not RECOVERY_HISTORY_FILE.exists():
        return {}

    try:
        payload = json.loads(
            RECOVERY_HISTORY_FILE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _save_recovery_history_unlocked(history):
    temporary = RECOVERY_HISTORY_FILE.with_suffix(
        RECOVERY_HISTORY_FILE.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(RECOVERY_HISTORY_FILE)


def get_recovery_history():
    with _history_lock:
        return _load_recovery_history_unlocked()


def reset_recovery_history():
    with _history_lock:
        _save_recovery_history_unlocked({})


def get_recovery_confidence(decision):
    with _history_lock:
        history = _load_recovery_history_unlocked()
        record = history.get(decision, {})
        successes = int(record.get("successes", 0))
        executions = int(record.get("executions", 0))

    denominator = (
        executions
        + RECOVERY_PRIOR_ALPHA
        + RECOVERY_PRIOR_BETA
    )
    if denominator <= 0:
        return 0.5

    return clamp(
        (successes + RECOVERY_PRIOR_ALPHA)
        / denominator
    )


def update_recovery_history(decision, success):
    with _history_lock:
        history = _load_recovery_history_unlocked()
        record = history.setdefault(
            decision,
            {"successes": 0, "executions": 0},
        )
        record["executions"] = int(record.get("executions", 0)) + 1
        if success:
            record["successes"] = int(record.get("successes", 0)) + 1
        record["last_updated"] = datetime.now(timezone.utc).isoformat()
        _save_recovery_history_unlocked(history)


def calculate_hdcm_confidence(candidate):
    return (
        max(candidate["ce"], 1e-9) ** HDCM_WEIGHTS["ce"]
        * max(candidate["ch"], 1e-9) ** HDCM_WEIGHTS["ch"]
        * max(candidate["cc"], 1e-9) ** HDCM_WEIGHTS["cc"]
        * max(candidate["cr"], 1e-9) ** HDCM_WEIGHTS["cr"]
    )


def calculate_additive_confidence(candidate):
    return clamp(
        HDCM_WEIGHTS["ce"] * candidate["ce"]
        + HDCM_WEIGHTS["ch"] * candidate["ch"]
        + HDCM_WEIGHTS["cc"] * candidate["cc"]
        + HDCM_WEIGHTS["cr"] * candidate["cr"]
    )


def rank_candidates(candidates, aggregation_mode=None):
    ranked = []
    mode = (aggregation_mode or HDCM_AGGREGATION_MODE).lower()
    if mode not in {"multiplicative", "additive"}:
        mode = "multiplicative"

    for decision, values in candidates.items():
        candidate = {
            "decision": decision,
            "ce": values["ce"],
            "ch": values["ch"],
            "cc": values["cc"],
            "cr": get_recovery_confidence(decision),
            "aggregation_mode": mode,
        }

        if mode == "additive":
            candidate["confidence"] = calculate_additive_confidence(candidate)
        else:
            candidate["confidence"] = calculate_hdcm_confidence(candidate)

        ranked.append(candidate)

    return sorted(
        ranked,
        key=lambda item: item["confidence"],
        reverse=True,
    )


def evaluate_safety_gate(ranked):
    selected = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"confidence": 0.0}
    margin = selected["confidence"] - second["confidence"]
    reasons = []

    if selected["decision"] not in AUTONOMOUS_ACTION_ALLOWLIST:
        reasons.append("selected action is not on the autonomous-action allowlist")
    if selected["confidence"] < MIN_AUTONOMOUS_CONFIDENCE:
        reasons.append(
            f"confidence {selected['confidence']:.3f} is below "
            f"the autonomous threshold {MIN_AUTONOMOUS_CONFIDENCE:.3f}"
        )
    if margin < MIN_CONFIDENCE_MARGIN:
        reasons.append(
            f"confidence margin {margin:.3f} is below "
            f"the ambiguity threshold {MIN_CONFIDENCE_MARGIN:.3f}"
        )

    return {
        "allowed": not reasons,
        "status": "autonomous" if not reasons else "operator_escalation",
        "reason": "; ".join(reasons),
        "confidence_margin": margin,
    }


# ============================================================
# Prometheus and application communication
# ============================================================

def prometheus_query(query):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()
    results = payload["data"]["result"]

    if not results:
        return 0.0

    value = results[0]["value"][1]

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def apply_fault(fault_type, severity):
    response = requests.post(
        f"{APP_URL}/fault",
        json={
            "type": fault_type,
            "severity": severity,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def clear_fault():
    return apply_fault(
        "clear",
        0,
    )


def generate_load(
    duration_seconds=10,
    requests_per_second=6,
):
    latencies = []
    successful_requests = 0
    failed_requests = 0

    started = time.time()

    while (
        time.time() - started
        < duration_seconds
    ):
        batch_started = time.time()

        for _ in range(requests_per_second):
            request_started = time.perf_counter()

            try:
                response = requests.get(
                    f"{APP_URL}/work",
                    timeout=5,
                )

                if response.status_code < 400:
                    successful_requests += 1
                else:
                    failed_requests += 1

            except requests.RequestException:
                failed_requests += 1

            latencies.append(
                (
                    time.perf_counter()
                    - request_started
                )
                * 1000
            )

        elapsed = time.time() - batch_started

        if elapsed < 1:
            time.sleep(1 - elapsed)

    total_requests = (
        successful_requests
        + failed_requests
    )

    availability = (
        successful_requests / total_requests
        if total_requests > 0
        else 0.0
    )

    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "availability": availability,
        "median_client_latency_ms": (
            statistics.median(latencies)
            if latencies
            else 0.0
        ),
    }


def collect_observation():
    cpu = prometheus_query(
        "ecdm_cpu_usage_percent"
    )

    memory = prometheus_query(
        "ecdm_memory_usage_percent"
    )

    allocated_memory_mb = prometheus_query(
        "ecdm_memory_allocated_mb"
    )

    error_rate = prometheus_query(
        "ecdm_error_rate_percent"
    )

    dependency_available = prometheus_query(
        "ecdm_dependency_available"
    )

    latency_seconds = prometheus_query(
        "rate(ecdm_request_latency_seconds_sum[1m])"
        " / "
        "clamp_min("
        "rate(ecdm_request_latency_seconds_count[1m]), "
        "0.001)"
    )

    return {
        "cpu": cpu,
        "memory": memory,
        "allocated_memory_mb": (
            allocated_memory_mb
        ),
        "error_rate": error_rate,
        "dependency_available": (
            dependency_available
        ),
        "latency_ms": latency_seconds * 1000,
    }


# ============================================================
# Recovery execution
# ============================================================

SCENARIO_FAULT_KEYS = {
    "cpu": "cpu_load",
    "memory": "memory_mb",
    "dependency": "dependency_failure",
    "latency": "latency_ms",
    "errors": "error_probability",
}


def _fault_is_cleared(faults, scenario_key):
    key = SCENARIO_FAULT_KEYS.get(scenario_key)
    if key is None:
        return True

    value = faults.get(key)
    if key == "dependency_failure":
        return not bool(value)

    try:
        return abs(float(value or 0)) < 1e-12
    except (TypeError, ValueError):
        return False


def verify_recovery(scenario_key):
    attempts = 0
    consecutive = 0

    for attempt in range(1, RECOVERY_VERIFICATION_MAX_ATTEMPTS + 1):
        attempts = attempt
        response_ok = False
        fault_cleared = False

        try:
            health_response = requests.get(
                f"{APP_URL}/health",
                timeout=5,
            )
            health_response.raise_for_status()
            faults = health_response.json().get("faults", {})
            fault_cleared = _fault_is_cleared(faults, scenario_key)

            work_response = requests.get(
                f"{APP_URL}/work",
                timeout=5,
            )
            response_ok = work_response.status_code < 400
        except (requests.RequestException, ValueError):
            response_ok = False
            fault_cleared = False

        if response_ok and fault_cleared:
            consecutive += 1
            if consecutive >= RECOVERY_REQUIRED_CONSECUTIVE_CHECKS:
                return {
                    "success": True,
                    "attempts": attempts,
                    "consecutive_checks": consecutive,
                }
        else:
            consecutive = 0

        if attempt < RECOVERY_VERIFICATION_MAX_ATTEMPTS:
            time.sleep(RECOVERY_VERIFICATION_INTERVAL_SECONDS)

    return {
        "success": False,
        "attempts": attempts,
        "consecutive_checks": consecutive,
    }


def apply_recovery_action(decision):
    if ACTION_EXECUTION_MODE != "prototype_fault_reset":
        raise RuntimeError(
            "Only prototype_fault_reset execution mode is implemented in this "
            "research prototype. Real restart/rollback/scale-out operations "
            "must be provided by an infrastructure-specific adapter."
        )

    if decision == "Do Nothing":
        return "No recovery action was executed."

    if decision == "Scale Out":
        clear_fault()
        return (
            "Prototype action: the injected CPU-pressure state was cleared, "
            "representing restored processing capacity."
        )

    if decision == "Restart Service":
        clear_fault()
        return (
            "Prototype action: the application fault state was reset, "
            "representing service restart semantics."
        )

    if decision == "Restart Dependency":
        clear_fault()
        return (
            "Prototype action: the injected dependency failure was cleared, "
            "representing dependency restart semantics."
        )

    if decision == "Circuit Breaker":
        clear_fault()
        return (
            "Prototype action: the injected dependency failure was cleared, "
            "representing isolation/recovery semantics."
        )

    if decision == "Rollback":
        clear_fault()
        return (
            "Prototype action: the injected runtime fault was cleared, "
            "representing rollback to a stable configuration."
        )

    raise ValueError(f"Unsupported recovery decision: {decision}")


def build_safety_hold_result(safety):
    return {
        "success": False,
        "recovery_attempted": False,
        "executed_decision": "Operator Review",
        "operator_escalation": True,
        "safety_status": safety["status"],
        "safety_reason": safety["reason"],
        "action_execution_mode": ACTION_EXECUTION_MODE,
        "recovery_verification_latency_ms": 0.0,
        "verification_attempts": 0,
        "description": (
            "Autonomous execution was withheld by the safety gate. "
            + safety["reason"]
        ),
    }


def execute_recovery(decision, scenario_key, safety):
    if not safety["allowed"]:
        return build_safety_hold_result(safety)

    recovery_started = time.perf_counter()

    try:
        description = apply_recovery_action(decision)
        verification = verify_recovery(scenario_key)
        success = verification["success"]
        attempts = verification["attempts"]
    except Exception as error:
        success = False
        attempts = 0
        description = f"Recovery action failed before verification: {error}"

    recovery_verification_latency_ms = (
        time.perf_counter() - recovery_started
    ) * 1000

    update_recovery_history(decision, success)

    return {
        "success": success,
        "recovery_attempted": True,
        "executed_decision": decision,
        "operator_escalation": not success,
        "safety_status": (
            "autonomous" if success else "operator_escalation_after_failure"
        ),
        "safety_reason": (
            "" if success else "selected recovery action did not satisfy the recovery verification criteria"
        ),
        "action_execution_mode": ACTION_EXECUTION_MODE,
        "recovery_verification_latency_ms": recovery_verification_latency_ms,
        "verification_attempts": attempts,
        "description": description,
    }


def execute_ranked_recovery(ranked, scenario_key):
    selected = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"confidence": 0.0}
    safety = evaluate_safety_gate(ranked)
    recovery = execute_recovery(
        selected["decision"],
        scenario_key,
        safety,
    )
    return selected, second, safety, recovery


# ============================================================
# CSV storage
# ============================================================

CSV_FIELDS = [
    "timestamp",
    "scenario",
    "cpu",
    "memory",
    "allocated_memory_mb",
    "latency_ms",
    "error_rate",
    "dependency_available",
    "selected_decision",
    "executed_decision",
    "aggregation_mode",
    "ce",
    "ch",
    "cc",
    "cr",
    "confidence",
    "confidence_margin",
    "correct",
    "decision_latency_ms",
    "recovery_success",
    "recovery_attempted",
    "operator_escalation",
    "safety_status",
    "safety_reason",
    "action_execution_mode",
    "recovery_verification_latency_ms",
    "verification_attempts",
    "pre_availability",
    "post_availability",
    "availability_gain",
    "explanation",
]


def save_result(row):
    file_exists = DATA_FILE.exists()

    with DATA_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        complete_row = {
            field: row.get(field, "")
            for field in CSV_FIELDS
        }

        writer.writerow(complete_row)


def build_result_row(
    scenario,
    observation,
    selected,
    second,
    expected_decision,
    decision_latency_ms,
    recovery,
    pre_load,
    post_load,
    explanation,
):
    return {
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "scenario": scenario,
        "cpu": round(
            observation.get("cpu", 0),
            4,
        ),
        "memory": round(
            observation.get("memory", 0),
            4,
        ),
        "allocated_memory_mb": round(
            observation.get(
                "allocated_memory_mb",
                0,
            ),
            4,
        ),
        "latency_ms": round(
            observation.get(
                "latency_ms",
                0,
            ),
            4,
        ),
        "error_rate": round(
            observation.get(
                "error_rate",
                0,
            ),
            4,
        ),
        "dependency_available": round(
            observation.get(
                "dependency_available",
                0,
            ),
            4,
        ),
        "selected_decision": (
            selected["decision"]
        ),
        "executed_decision": recovery.get("executed_decision", ""),
        "aggregation_mode": selected.get("aggregation_mode", HDCM_AGGREGATION_MODE),
        "ce": round(selected["ce"], 6),
        "ch": round(selected["ch"], 6),
        "cc": round(selected["cc"], 6),
        "cr": round(selected["cr"], 6),
        "confidence": round(
            selected["confidence"],
            6,
        ),
        "confidence_margin": round(
            selected["confidence"]
            - second["confidence"],
            6,
        ),
        "correct": int(
            selected["decision"]
            == expected_decision
        ),
        "decision_latency_ms": round(
            decision_latency_ms,
            6,
        ),
        "recovery_success": int(recovery["success"]),
        "recovery_attempted": int(recovery.get("recovery_attempted", False)),
        "operator_escalation": int(recovery.get("operator_escalation", False)),
        "safety_status": recovery.get("safety_status", ""),
        "safety_reason": recovery.get("safety_reason", ""),
        "action_execution_mode": recovery.get("action_execution_mode", ACTION_EXECUTION_MODE),
        "recovery_verification_latency_ms": round(
            recovery.get("recovery_verification_latency_ms", 0.0),
            4,
        ),
        "verification_attempts": (
            recovery[
                "verification_attempts"
            ]
        ),
        "pre_availability": round(
            pre_load["availability"],
            6,
        ),
        "post_availability": round(
            post_load["availability"],
            6,
        ),
        "availability_gain": round(
            post_load["availability"]
            - pre_load["availability"],
            6,
        ),
        "explanation": explanation,
    }


# ============================================================
# CPU scenario
# ============================================================

def calculate_cpu_scenario_scores(
    observation,
):
    cpu_signal = clamp(
        (observation["cpu"] - 50) / 50
    )

    latency_signal = clamp(
        (
            observation["latency_ms"] - 100
        )
        / 1900
    )

    memory_signal = clamp(
        (
            observation["memory"] - 60
        )
        / 40
    )

    error_signal = clamp(
        observation["error_rate"] / 20
    )

    candidates = {
        "Scale Out": {
            "ce": clamp(
                0.75 * cpu_signal
                + 0.25 * latency_signal
            ),
            "ch": clamp(
                0.70 * cpu_signal
                + 0.30 * latency_signal
            ),
            "cc": 0.95,
        },
        "Restart Service": {
            "ce": clamp(
                0.45 * cpu_signal
                + 0.30 * memory_signal
                + 0.25 * error_signal
            ),
            "ch": clamp(
                0.55 * cpu_signal
                + 0.45 * error_signal
            ),
            "cc": 0.80,
        },
        "Do Nothing": {
            "ce": clamp(
                1
                - max(
                    cpu_signal,
                    latency_signal,
                    error_signal,
                )
            ),
            "ch": clamp(
                1 - cpu_signal
            ),
            "cc": 0.45,
        },
    }

    return rank_candidates(candidates)


@app.post("/run/cpu")
def run_cpu_scenario():
    payload = request.get_json(silent=True) or {}
    severity = float(payload.get("severity", 0.95))

    clear_fault()
    time.sleep(2)

    apply_fault(
        "cpu",
        severity,
    )

    pre_load = generate_load(
        duration_seconds=12,
        requests_per_second=8,
    )

    time.sleep(3)

    observation = collect_observation()

    decision_started = time.perf_counter()

    ranked = calculate_cpu_scenario_scores(
        observation
    )

    decision_latency_ms = (
        time.perf_counter()
        - decision_started
    ) * 1000

    selected, second, safety, recovery = execute_ranked_recovery(
        ranked,
        "cpu",
    )

    post_load = generate_load(
        duration_seconds=5,
        requests_per_second=5,
    )

    explanation = (
        f"{selected['decision']} was selected "
        f"with confidence "
        f"{selected['confidence']:.3f}. "
        f"CPU usage was "
        f"{observation['cpu']:.2f}% and "
        f"latency was "
        f"{observation['latency_ms']:.2f} ms. "
        f"The confidence margin over "
        f"{second['decision']} was "
        f"{selected['confidence'] - second['confidence']:.3f}."
    )

    row = build_result_row(
        scenario="CPU Resource Exhaustion",
        observation=observation,
        selected=selected,
        second=second,
        expected_decision="Scale Out",
        decision_latency_ms=decision_latency_ms,
        recovery=recovery,
        pre_load=pre_load,
        post_load=post_load,
        explanation=explanation,
    )

    save_result(row)

    return jsonify(
        status="completed",
        scenario="CPU Resource Exhaustion",
        observation=observation,
        ranked_decisions=ranked,
        selected_decision=selected,
        decision_latency_ms=(
            decision_latency_ms
        ),
        recovery=recovery,
        safety=safety,
        pre_recovery=pre_load,
        post_recovery=post_load,
        explanation=explanation,
        csv_file=str(DATA_FILE),
    )


# ============================================================
# Memory scenario
# ============================================================

def calculate_memory_scenario_scores(
    observation,
):
    allocated_signal = clamp(
        observation.get(
            "allocated_memory_mb",
            0,
        )
        / 300
    )

    memory_signal = clamp(
        observation["memory"] / 20
    )

    latency_signal = clamp(
        (
            observation["latency_ms"] - 100
        )
        / 1900
    )

    error_signal = clamp(
        observation["error_rate"] / 20
    )

    candidates = {
        "Restart Service": {
            "ce": clamp(
                0.70 * allocated_signal
                + 0.20 * memory_signal
                + 0.10 * latency_signal
            ),
            "ch": clamp(
                0.75 * allocated_signal
                + 0.15 * memory_signal
                + 0.10 * latency_signal
            ),
            "cc": 0.92,
        },
        "Scale Out": {
            "ce": clamp(
                0.45 * allocated_signal
                + 0.30 * memory_signal
                + 0.25 * latency_signal
            ),
            "ch": clamp(
                0.40 * allocated_signal
                + 0.30 * memory_signal
                + 0.30 * latency_signal
            ),
            "cc": 0.76,
        },
        "Do Nothing": {
            "ce": clamp(
                1
                - max(
                    allocated_signal,
                    memory_signal,
                    latency_signal,
                    error_signal,
                )
            ),
            "ch": clamp(
                1 - allocated_signal
            ),
            "cc": 0.35,
        },
    }

    return rank_candidates(candidates)


@app.post("/run/memory")
def run_memory_scenario():
    payload = request.get_json(silent=True) or {}
    severity = float(payload.get("severity", 300))

    clear_fault()
    time.sleep(2)

    apply_fault(
        "memory",
        severity,
    )

    pre_load = generate_load(
        duration_seconds=10,
        requests_per_second=8,
    )

    time.sleep(3)

    observation = collect_observation()

    decision_started = time.perf_counter()

    ranked = (
        calculate_memory_scenario_scores(
            observation
        )
    )

    decision_latency_ms = (
        time.perf_counter()
        - decision_started
    ) * 1000

    selected, second, safety, recovery = execute_ranked_recovery(
        ranked,
        "memory",
    )

    post_load = generate_load(
        duration_seconds=5,
        requests_per_second=5,
    )

    explanation = (
        f"{selected['decision']} was selected "
        f"with confidence "
        f"{selected['confidence']:.3f}. "
        f"Injected memory was "
        f"{observation.get('allocated_memory_mb', 0):.2f} MB "
        f"and process memory usage was "
        f"{observation['memory']:.2f}%."
    )

    row = build_result_row(
        scenario="Memory Pressure",
        observation=observation,
        selected=selected,
        second=second,
        expected_decision=(
            "Restart Service"
        ),
        decision_latency_ms=decision_latency_ms,
        recovery=recovery,
        pre_load=pre_load,
        post_load=post_load,
        explanation=explanation,
    )

    save_result(row)

    return jsonify(
        status="completed",
        scenario="Memory Pressure",
        observation=observation,
        ranked_decisions=ranked,
        selected_decision=selected,
        decision_latency_ms=(
            decision_latency_ms
        ),
        recovery=recovery,
        safety=safety,
        pre_recovery=pre_load,
        post_recovery=post_load,
        explanation=explanation,
        csv_file=str(DATA_FILE),
    )


# ============================================================
# Dependency scenario
# ============================================================

def calculate_dependency_scenario_scores(
    observation,
):
    dependency_signal = clamp(
        1
        - observation.get(
            "dependency_available",
            1,
        )
    )

    error_signal = clamp(
        observation["error_rate"] / 20
    )

    latency_signal = clamp(
        (
            observation["latency_ms"] - 100
        )
        / 1900
    )

    cpu_signal = clamp(
        (
            observation["cpu"] - 60
        )
        / 40
    )

    candidates = {
        "Restart Dependency": {
            "ce": clamp(
                0.75 * dependency_signal
                + 0.15 * error_signal
                + 0.10 * latency_signal
            ),
            "ch": clamp(
                0.80 * dependency_signal
                + 0.20 * error_signal
            ),
            "cc": 0.94,
        },
        "Circuit Breaker": {
            "ce": clamp(
                0.70 * dependency_signal
                + 0.20 * error_signal
                + 0.10 * latency_signal
            ),
            "ch": clamp(
                0.75 * dependency_signal
                + 0.25 * error_signal
            ),
            "cc": 0.91,
        },
        "Scale Out": {
            "ce": clamp(
                0.20 * dependency_signal
                + 0.30 * error_signal
                + 0.30 * latency_signal
                + 0.20 * cpu_signal
            ),
            "ch": clamp(
                0.20 * dependency_signal
                + 0.40 * latency_signal
                + 0.40 * cpu_signal
            ),
            "cc": 0.55,
        },
        "Do Nothing": {
            "ce": clamp(
                1
                - max(
                    dependency_signal,
                    error_signal,
                    latency_signal,
                )
            ),
            "ch": clamp(
                1 - dependency_signal
            ),
            "cc": 0.25,
        },
    }

    return rank_candidates(candidates)


@app.post("/run/dependency")
def run_dependency_scenario():
    clear_fault()
    time.sleep(2)

    generate_load(
        duration_seconds=3,
        requests_per_second=4,
    )

    apply_fault(
        "dependency",
        1,
    )

    pre_load = generate_load(
        duration_seconds=10,
        requests_per_second=6,
    )

    time.sleep(3)

    observation = collect_observation()

    decision_started = time.perf_counter()

    ranked = (
        calculate_dependency_scenario_scores(
            observation
        )
    )

    decision_latency_ms = (
        time.perf_counter()
        - decision_started
    ) * 1000

    selected, second, safety, recovery = execute_ranked_recovery(
        ranked,
        "dependency",
    )

    post_load = generate_load(
        duration_seconds=5,
        requests_per_second=5,
    )

    explanation = (
        f"{selected['decision']} was selected "
        f"with confidence "
        f"{selected['confidence']:.3f}. "
        f"Dependency availability was "
        f"{observation.get('dependency_available', 0):.2f}, "
        f"error rate was "
        f"{observation['error_rate']:.2f}%, "
        f"and latency was "
        f"{observation['latency_ms']:.2f} ms. "
        f"The confidence margin over "
        f"{second['decision']} was "
        f"{selected['confidence'] - second['confidence']:.3f}."
    )

    row = build_result_row(
        scenario=(
            "Service Dependency Failure"
        ),
        observation=observation,
        selected=selected,
        second=second,
        expected_decision=(
            "Restart Dependency"
        ),
        decision_latency_ms=decision_latency_ms,
        recovery=recovery,
        pre_load=pre_load,
        post_load=post_load,
        explanation=explanation,
    )

    save_result(row)

    return jsonify(
        status="completed",
        scenario=(
            "Service Dependency Failure"
        ),
        observation=observation,
        ranked_decisions=ranked,
        selected_decision=selected,
        decision_latency_ms=(
            decision_latency_ms
        ),
        recovery=recovery,
        safety=safety,
        pre_recovery=pre_load,
        post_recovery=post_load,
        explanation=explanation,
        csv_file=str(DATA_FILE),
    )



# ============================================================
# Application latency scenario
# ============================================================

def calculate_latency_scenario_scores(
    observation,
):
    latency_signal = clamp(
        (
            observation["latency_ms"] - 100
        )
        / 1400
    )

    error_signal = clamp(
        observation["error_rate"] / 20
    )

    cpu_signal = clamp(
        (
            observation["cpu"] - 60
        )
        / 40
    )

    memory_signal = clamp(
        (
            observation["memory"] - 60
        )
        / 40
    )

    dependency_signal = clamp(
        1
        - observation.get(
            "dependency_available",
            1,
        )
    )

    candidates = {
        "Rollback": {
            "ce": clamp(
                0.75 * latency_signal
                + 0.10 * error_signal
                + 0.10 * (
                    1 - cpu_signal
                )
                + 0.05 * (
                    1 - dependency_signal
                )
            ),
            "ch": clamp(
                0.80 * latency_signal
                + 0.10 * error_signal
                + 0.10 * (
                    1 - dependency_signal
                )
            ),
            "cc": 0.94,
        },

        "Restart Service": {
            "ce": clamp(
                0.60 * latency_signal
                + 0.20 * error_signal
                + 0.10 * cpu_signal
                + 0.10 * memory_signal
            ),
            "ch": clamp(
                0.65 * latency_signal
                + 0.20 * error_signal
                + 0.15 * cpu_signal
            ),
            "cc": 0.84,
        },

        "Scale Out": {
            "ce": clamp(
                0.45 * latency_signal
                + 0.40 * cpu_signal
                + 0.15 * error_signal
            ),
            "ch": clamp(
                0.40 * latency_signal
                + 0.50 * cpu_signal
                + 0.10 * error_signal
            ),
            "cc": (
                0.80
                if cpu_signal >= 0.50
                else 0.58
            ),
        },

        "Do Nothing": {
            "ce": clamp(
                1
                - max(
                    latency_signal,
                    error_signal,
                )
            ),
            "ch": clamp(
                1 - latency_signal
            ),
            "cc": 0.25,
        },
    }

    return rank_candidates(candidates)


@app.post("/run/latency")
def run_latency_scenario():
    payload = request.get_json(silent=True) or {}
    severity = float(payload.get("severity", 1200))

    clear_fault()
    time.sleep(2)

    # Establish a short normal baseline.
    generate_load(
        duration_seconds=3,
        requests_per_second=4,
    )

    # Add 1200 ms delay to every application request.
    apply_fault(
        "latency",
        severity,
    )

    pre_load = generate_load(
        duration_seconds=10,
        requests_per_second=4,
    )

    time.sleep(3)

    observation = collect_observation()

    decision_started = time.perf_counter()

    ranked = (
        calculate_latency_scenario_scores(
            observation
        )
    )

    decision_latency_ms = (
        time.perf_counter()
        - decision_started
    ) * 1000

    selected, second, safety, recovery = execute_ranked_recovery(
        ranked,
        "latency",
    )

    post_load = generate_load(
        duration_seconds=5,
        requests_per_second=5,
    )

    explanation = (
        f"{selected['decision']} was selected "
        f"with confidence "
        f"{selected['confidence']:.3f}. "
        f"Observed latency reached "
        f"{observation['latency_ms']:.2f} ms, "
        f"while CPU usage was "
        f"{observation['cpu']:.2f}% and "
        f"dependency availability was "
        f"{observation.get('dependency_available', 0):.2f}. "
        f"The confidence margin over "
        f"{second['decision']} was "
        f"{selected['confidence'] - second['confidence']:.3f}."
    )

    row = build_result_row(
        scenario="Application Latency",
        observation=observation,
        selected=selected,
        second=second,
        expected_decision="Rollback",
        decision_latency_ms=decision_latency_ms,
        recovery=recovery,
        pre_load=pre_load,
        post_load=post_load,
        explanation=explanation,
    )

    save_result(row)

    return jsonify(
        status="completed",
        scenario="Application Latency",
        observation=observation,
        ranked_decisions=ranked,
        selected_decision=selected,
        decision_latency_ms=(
            decision_latency_ms
        ),
        recovery=recovery,
        safety=safety,
        pre_recovery=pre_load,
        post_recovery=post_load,
        explanation=explanation,
        csv_file=str(DATA_FILE),
    )



# ============================================================
# Application error injection scenario
# ============================================================

def calculate_error_scenario_scores(
    observation,
):
    error_signal = clamp(
        observation["error_rate"] / 40
    )

    latency_signal = clamp(
        (
            observation["latency_ms"] - 100
        )
        / 1900
    )

    cpu_signal = clamp(
        (
            observation["cpu"] - 60
        )
        / 40
    )

    memory_signal = clamp(
        (
            observation["memory"] - 60
        )
        / 40
    )

    dependency_signal = clamp(
        1
        - observation.get(
            "dependency_available",
            1,
        )
    )

    candidates = {
        "Rollback": {
            "ce": clamp(
                0.75 * error_signal
                + 0.10 * latency_signal
                + 0.10 * (
                    1 - dependency_signal
                )
                + 0.05 * (
                    1 - cpu_signal
                )
            ),
            "ch": clamp(
                0.80 * error_signal
                + 0.10 * latency_signal
                + 0.10 * (
                    1 - dependency_signal
                )
            ),
            "cc": 0.95,
        },

        "Restart Service": {
            "ce": clamp(
                0.60 * error_signal
                + 0.20 * latency_signal
                + 0.10 * cpu_signal
                + 0.10 * memory_signal
            ),
            "ch": clamp(
                0.65 * error_signal
                + 0.20 * latency_signal
                + 0.15 * cpu_signal
            ),
            "cc": 0.86,
        },

        "Scale Out": {
            "ce": clamp(
                0.35 * error_signal
                + 0.45 * cpu_signal
                + 0.20 * latency_signal
            ),
            "ch": clamp(
                0.30 * error_signal
                + 0.50 * cpu_signal
                + 0.20 * latency_signal
            ),
            "cc": (
                0.78
                if cpu_signal >= 0.50
                else 0.55
            ),
        },

        "Do Nothing": {
            "ce": clamp(
                1
                - max(
                    error_signal,
                    latency_signal,
                )
            ),
            "ch": clamp(
                1 - error_signal
            ),
            "cc": 0.20,
        },
    }

    return rank_candidates(candidates)


@app.post("/run/errors")
def run_error_scenario():
    payload = request.get_json(silent=True) or {}
    severity = float(payload.get("severity", 0.35))

    clear_fault()
    time.sleep(2)

    # Short normal baseline
    generate_load(
        duration_seconds=3,
        requests_per_second=4,
    )

    # Inject 35% application error probability
    apply_fault(
        "errors",
        severity,
    )

    pre_load = generate_load(
        duration_seconds=10,
        requests_per_second=8,
    )

    time.sleep(3)

    observation = collect_observation()

    decision_started = time.perf_counter()

    ranked = (
        calculate_error_scenario_scores(
            observation
        )
    )

    decision_latency_ms = (
        time.perf_counter()
        - decision_started
    ) * 1000

    selected, second, safety, recovery = execute_ranked_recovery(
        ranked,
        "errors",
    )

    post_load = generate_load(
        duration_seconds=5,
        requests_per_second=6,
    )

    explanation = (
        f"{selected['decision']} was selected "
        f"with confidence "
        f"{selected['confidence']:.3f}. "
        f"The observed error rate was "
        f"{observation['error_rate']:.2f}% "
        f"with latency of "
        f"{observation['latency_ms']:.2f} ms. "
        f"CPU usage was "
        f"{observation['cpu']:.2f}%, "
        f"and dependency availability was "
        f"{observation.get('dependency_available', 0):.2f}. "
        f"The confidence margin over "
        f"{second['decision']} was "
        f"{selected['confidence'] - second['confidence']:.3f}."
    )

    row = build_result_row(
        scenario="Application Error Injection",
        observation=observation,
        selected=selected,
        second=second,
        expected_decision="Rollback",
        decision_latency_ms=decision_latency_ms,
        recovery=recovery,
        pre_load=pre_load,
        post_load=post_load,
        explanation=explanation,
    )

    save_result(row)

    return jsonify(
        status="completed",
        scenario="Application Error Injection",
        observation=observation,
        ranked_decisions=ranked,
        selected_decision=selected,
        decision_latency_ms=(
            decision_latency_ms
        ),
        recovery=recovery,
        safety=safety,
        pre_recovery=pre_load,
        post_recovery=post_load,
        explanation=explanation,
        csv_file=str(DATA_FILE),
    )



# ============================================================
# Batch experiment execution
# ============================================================

SCENARIO_ENDPOINTS = {
    "cpu": run_cpu_scenario,
    "memory": run_memory_scenario,
    "dependency": run_dependency_scenario,
    "latency": run_latency_scenario,
    "errors": run_error_scenario,
}


@app.post("/run/batch")
def run_batch_experiments():
    from flask import request

    payload = request.get_json(silent=True) or {}

    repetitions = int(
        payload.get(
            "repetitions",
            3,
        )
    )

    repetitions = max(
        1,
        min(repetitions, 50),
    )

    requested_scenarios = payload.get(
        "scenarios",
        list(SCENARIO_ENDPOINTS.keys()),
    )

    selected_scenarios = [
        name
        for name in requested_scenarios
        if name in SCENARIO_ENDPOINTS
    ]

    if not selected_scenarios:
        return jsonify(
            status="error",
            message="No valid scenarios were selected.",
        ), 400

    started = time.time()
    completed = []
    failures = []

    for repetition in range(
        1,
        repetitions + 1,
    ):
        for scenario_name in selected_scenarios:
            try:
                response = (
                    SCENARIO_ENDPOINTS[
                        scenario_name
                    ]()
                )

                if isinstance(response, tuple):
                    flask_response = response[0]
                else:
                    flask_response = response

                payload_data = (
                    flask_response.get_json()
                )

                completed.append(
                    {
                        "repetition": repetition,
                        "scenario": scenario_name,
                        "selected_decision": (
                            payload_data.get(
                                "selected_decision",
    "executed_decision",
    "aggregation_mode",
                                {},
                            ).get(
                                "decision",
                                "",
                            )
                        ),
                        "confidence": (
                            payload_data.get(
                                "selected_decision",
    "executed_decision",
    "aggregation_mode",
                                {},
                            ).get(
                                "confidence",
                                0,
                            )
                        ),
                        "recovery_success": (
                            payload_data.get(
                                "recovery",
                                {},
                            ).get(
                                "success",
                                False,
                            )
                        ),
                        "recovery_verification_latency_ms": (
                            payload_data.get(
                                "recovery",
                                {},
                            ).get(
                                "recovery_verification_latency_ms",
                                0,
                            )
                        ),
                    }
                )

            except Exception as error:
                failures.append(
                    {
                        "repetition": repetition,
                        "scenario": scenario_name,
                        "error": str(error),
                    }
                )

                try:
                    clear_fault()
                except Exception:
                    pass

    elapsed = time.time() - started

    return jsonify(
        status="completed",
        repetitions_by_scenario={
            scenario: repetitions_by_scenario[scenario]
            for scenario in selected_scenarios
        },
        scenarios=selected_scenarios,
        expected_episodes=(
            repetitions
            * len(selected_scenarios)
        ),
        completed_episodes=len(completed),
        failed_episodes=len(failures),
        elapsed_seconds=elapsed,
        completed=completed,
        failures=failures,
        csv_file=str(DATA_FILE),
    )



# ============================================================
# Multi-severity experimental campaign
# ============================================================

import uuid


CAMPAIGN_FILE = DATA_FILE.parent / "campaign_results_revised.csv"

campaign_state = {
    "status": "idle",
    "campaign_id": None,
    "total_episodes": 0,
    "completed_episodes": 0,
    "failed_episodes": 0,
    "current_scenario": None,
    "current_severity": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
}


CAMPAIGN_LEVELS = {
    "cpu": [
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ],
    "memory": [
        100,
        150,
        200,
        250,
        300,
    ],
    "latency": [
        300,
        500,
        700,
        900,
        1200,
    ],
    "errors": [
        0.10,
        0.20,
        0.30,
        0.35,
        0.40,
    ],
    "dependency": [
        1.0,
    ],
}


CAMPAIGN_ENDPOINTS = {
    "cpu": "/run/cpu",
    "memory": "/run/memory",
    "dependency": "/run/dependency",
    "latency": "/run/latency",
    "errors": "/run/errors",
}


CAMPAIGN_FIELDS = [
    "campaign_id",
    "timestamp",
    "repetition",
    "scenario",
    "severity",
    "selected_decision",
    "executed_decision",
    "aggregation_mode",
    "expected_decision",
    "correct",
    "confidence",
    "confidence_margin",
    "ce",
    "ch",
    "cc",
    "cr",
    "decision_latency_ms",
    "recovery_success",
    "recovery_attempted",
    "operator_escalation",
    "safety_status",
    "safety_reason",
    "action_execution_mode",
    "recovery_verification_latency_ms",
    "pre_availability",
    "post_availability",
    "availability_gain",
    "cpu",
    "memory",
    "allocated_memory_mb",
    "latency_ms",
    "error_rate",
    "dependency_available",
]


EXPECTED_DECISIONS = {
    "cpu": "Scale Out",
    "memory": "Restart Service",
    "dependency": "Restart Dependency",
    "latency": "Rollback",
    "errors": "Rollback",
}


def save_campaign_result(row):
    file_exists = CAMPAIGN_FILE.exists()

    with CAMPAIGN_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CAMPAIGN_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            field: row.get(field, "")
            for field in CAMPAIGN_FIELDS
        })


def execute_campaign(
    campaign_id,
    repetitions_by_scenario,
    selected_scenarios,
):
    campaign_state.update({
        "status": "running",
        "campaign_id": campaign_id,
        "completed_episodes": 0,
        "failed_episodes": 0,
        "started_at": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "finished_at": None,
        "last_error": None,
    })

    combinations = []

    for scenario in selected_scenarios:
        scenario_repetitions = repetitions_by_scenario[scenario]
        for severity in CAMPAIGN_LEVELS[scenario]:
            for repetition in range(
                1,
                scenario_repetitions + 1,
            ):
                combinations.append(
                    (
                        scenario,
                        severity,
                        repetition,
                    )
                )

    campaign_state["total_episodes"] = len(
        combinations
    )

    for scenario, severity, repetition in combinations:
        campaign_state["current_scenario"] = (
            scenario
        )
        campaign_state["current_severity"] = (
            severity
        )

        try:
            response = requests.post(
                "http://127.0.0.1:8100"
                + CAMPAIGN_ENDPOINTS[scenario],
                json={
                    "severity": severity,
                },
                timeout=180,
            )

            response.raise_for_status()
            result = response.json()

            selected = result[
                "selected_decision"
            ]

            recovery = result["recovery"]
            observation = result["observation"]
            pre = result["pre_recovery"]
            post = result["post_recovery"]

            expected = EXPECTED_DECISIONS[
                scenario
            ]

            ranked = result[
                "ranked_decisions"
            ]

            second_confidence = (
                ranked[1]["confidence"]
                if len(ranked) > 1
                else 0
            )

            row = {
                "campaign_id": campaign_id,
                "timestamp": (
                    datetime.now(timezone.utc)
                    .isoformat()
                ),
                "repetition": repetition,
                "scenario": scenario,
                "severity": severity,
                "selected_decision": (
                    selected["decision"]
                ),
                "executed_decision": recovery.get("executed_decision", ""),
                "aggregation_mode": selected.get("aggregation_mode", HDCM_AGGREGATION_MODE),
                "expected_decision": expected,
                "correct": int(
                    selected["decision"]
                    == expected
                ),
                "confidence": (
                    selected["confidence"]
                ),
                "confidence_margin": (
                    selected["confidence"]
                    - second_confidence
                ),
                "ce": selected.get("ce", 0),
                "ch": selected.get("ch", 0),
                "cc": selected.get("cc", 0),
                "cr": selected.get("cr", 0),
                "decision_latency_ms": (
                    result.get(
                        "decision_latency_ms",
                        0,
                    )
                ),
                "recovery_success": int(
                    recovery["success"]
                ),
                "recovery_attempted": int(recovery.get("recovery_attempted", False)),
                "operator_escalation": int(recovery.get("operator_escalation", False)),
                "safety_status": recovery.get("safety_status", ""),
                "safety_reason": recovery.get("safety_reason", ""),
                "action_execution_mode": recovery.get("action_execution_mode", ACTION_EXECUTION_MODE),
                "recovery_verification_latency_ms": recovery.get(
                    "recovery_verification_latency_ms", 0
                ),
                "pre_availability": (
                    pre["availability"]
                ),
                "post_availability": (
                    post["availability"]
                ),
                "availability_gain": (
                    post["availability"]
                    - pre["availability"]
                ),
                "cpu": observation.get(
                    "cpu",
                    0,
                ),
                "memory": observation.get(
                    "memory",
                    0,
                ),
                "allocated_memory_mb": (
                    observation.get(
                        "allocated_memory_mb",
                        0,
                    )
                ),
                "latency_ms": (
                    observation.get(
                        "latency_ms",
                        0,
                    )
                ),
                "error_rate": (
                    observation.get(
                        "error_rate",
                        0,
                    )
                ),
                "dependency_available": (
                    observation.get(
                        "dependency_available",
                        0,
                    )
                ),
            }

            save_campaign_result(row)

            campaign_state[
                "completed_episodes"
            ] += 1

        except Exception as error:
            campaign_state[
                "failed_episodes"
            ] += 1

            campaign_state["last_error"] = (
                str(error)
            )

            try:
                clear_fault()
            except Exception:
                pass

    campaign_state.update({
        "status": "completed",
        "current_scenario": None,
        "current_severity": None,
        "finished_at": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
    })


DEFAULT_REPETITIONS_BY_SCENARIO = {
    "cpu": 6,
    "memory": 5,
    "dependency": 5,
    "latency": 5,
    "errors": 5,
}


@app.post("/campaign/start")
def start_campaign():
    if campaign_state["status"] == "running":
        return jsonify(
            status="error",
            message=(
                "An experimental campaign "
                "is already running."
            ),
            campaign=campaign_state,
        ), 409

    payload = request.get_json(
        silent=True
    ) or {}

    scalar_repetitions = payload.get("repetitions")
    requested_repetitions = payload.get("repetitions_by_scenario", {})

    repetitions_by_scenario = {}
    for scenario, default_value in DEFAULT_REPETITIONS_BY_SCENARIO.items():
        if scalar_repetitions is not None:
            value = int(scalar_repetitions)
        else:
            value = int(requested_repetitions.get(scenario, default_value))
        repetitions_by_scenario[scenario] = max(1, min(value, 20))

    selected_scenarios = payload.get(
        "scenarios",
        list(CAMPAIGN_LEVELS.keys()),
    )

    selected_scenarios = [
        scenario
        for scenario in selected_scenarios
        if scenario in CAMPAIGN_LEVELS
    ]

    if not selected_scenarios:
        return jsonify(
            status="error",
            message="No valid scenarios selected.",
        ), 400

    reset_history = bool(payload.get("reset_history", True))
    if reset_history:
        reset_recovery_history()

    campaign_id = str(uuid.uuid4())

    total = sum(
        len(CAMPAIGN_LEVELS[scenario])
        * repetitions_by_scenario[scenario]
        for scenario in selected_scenarios
    )

    campaign_state.update({
        "status": "starting",
        "campaign_id": campaign_id,
        "total_episodes": total,
        "completed_episodes": 0,
        "failed_episodes": 0,
    })

    thread = threading.Thread(
        target=execute_campaign,
        args=(
            campaign_id,
            repetitions_by_scenario,
            selected_scenarios,
        ),
        daemon=True,
    )

    thread.start()

    return jsonify(
        status="started",
        campaign_id=campaign_id,
        repetitions_by_scenario={
            scenario: repetitions_by_scenario[scenario]
            for scenario in selected_scenarios
        },
        scenarios=selected_scenarios,
        total_episodes=total,
        recovery_history_reset=reset_history,
        initial_recovery_confidence=(
            RECOVERY_PRIOR_ALPHA / (RECOVERY_PRIOR_ALPHA + RECOVERY_PRIOR_BETA)
            if RECOVERY_PRIOR_ALPHA + RECOVERY_PRIOR_BETA > 0
            else 0.5
        ),
        campaign_file=str(
            CAMPAIGN_FILE
        ),
    )


@app.get("/campaign/status")
def campaign_status():
    return jsonify(campaign_state)


@app.get("/campaign/results")
def campaign_results():
    if not CAMPAIGN_FILE.exists():
        return jsonify(
            status="empty",
            count=0,
            results=[],
        )

    with CAMPAIGN_FILE.open(
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    requested_campaign_id = (
        request.args.get("campaign_id")
    )

    if requested_campaign_id:
        rows = [
            row
            for row in rows
            if row["campaign_id"]
            == requested_campaign_id
        ]

    return jsonify(
        status="ok",
        count=len(rows),
        results=rows,
    )


@app.get("/history")
def recovery_history():
    return jsonify(
        status="ok",
        prior_alpha=RECOVERY_PRIOR_ALPHA,
        prior_beta=RECOVERY_PRIOR_BETA,
        history=get_recovery_history(),
    )


@app.post("/history/reset")
def recovery_history_reset():
    reset_recovery_history()
    return jsonify(
        status="reset",
        initial_recovery_confidence=(
            RECOVERY_PRIOR_ALPHA / (RECOVERY_PRIOR_ALPHA + RECOVERY_PRIOR_BETA)
            if RECOVERY_PRIOR_ALPHA + RECOVERY_PRIOR_BETA > 0
            else 0.5
        ),
    )


# ============================================================
# General API routes
# ============================================================

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        service="ECDM Engine",
        data_file=str(DATA_FILE),
        campaign_file=str(CAMPAIGN_FILE),
        recovery_history_file=str(RECOVERY_HISTORY_FILE),
        hdcm_weights=HDCM_WEIGHTS,
        hdcm_aggregation_mode=HDCM_AGGREGATION_MODE,
        minimum_autonomous_confidence=MIN_AUTONOMOUS_CONFIDENCE,
        minimum_confidence_margin=MIN_CONFIDENCE_MARGIN,
        action_execution_mode=ACTION_EXECUTION_MODE,
    )


@app.get("/results")
def results():
    if not DATA_FILE.exists():
        return jsonify(
            status="empty",
            count=0,
            results=[],
        )

    with DATA_FILE.open(
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    return jsonify(
        status="ok",
        count=len(rows),
        results=rows,
    )


@app.get("/routes")
def routes():
    return jsonify(
        routes=[
            "/health",
            "/results",
            "/routes",
            "/run/cpu",
            "/run/memory",
            "/run/dependency",
            "/run/latency",
            "/run/errors",
            "/campaign/start",
            "/campaign/status",
            "/campaign/results",
            "/history",
            "/history/reset",
        ]
    )


# This block must remain at the very end of the file.
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8100,
        threaded=True,
    )
