import csv
import os
import statistics
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


# ============================================================
# General utility functions
# ============================================================

def clamp(value):
    return max(
        0.0,
        min(1.0, float(value)),
    )


def calculate_hdcm_confidence(candidate):
    return (
        max(candidate["ce"], 1e-9) ** 0.25
        * max(candidate["ch"], 1e-9) ** 0.25
        * max(candidate["cc"], 1e-9) ** 0.25
        * max(candidate["cr"], 1e-9) ** 0.25
    )


def rank_candidates(candidates):
    ranked = []

    for decision, values in candidates.items():
        candidate = {
            "decision": decision,
            "ce": values["ce"],
            "ch": values["ch"],
            "cc": values["cc"],
            "cr": values["cr"],
        }

        candidate["confidence"] = (
            calculate_hdcm_confidence(candidate)
        )

        ranked.append(candidate)

    return sorted(
        ranked,
        key=lambda item: item["confidence"],
        reverse=True,
    )


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

def execute_recovery(decision):
    recovery_started = time.perf_counter()

    if decision == "Scale Out":
        clear_fault()

        description = (
            "CPU pressure was removed, representing "
            "the addition of processing capacity."
        )

    elif decision == "Restart Service":
        clear_fault()

        description = (
            "The application runtime state and "
            "injected fault were reset."
        )

    elif decision == "Restart Dependency":
        clear_fault()

        description = (
            "The dependency failure was cleared and "
            "service communication was restored."
        )

    elif decision == "Circuit Breaker":
        clear_fault()

        description = (
            "The failed dependency was isolated and "
            "the request path was restored."
        )

    elif decision == "Rollback":
        clear_fault()

        description = (
            "The latency-producing runtime configuration "
            "was rolled back to the stable state."
        )

    elif decision == "Do Nothing":
        description = (
            "No recovery action was executed."
        )

    else:
        clear_fault()

        description = (
            "The fault state was reset using "
            "the default recovery procedure."
        )

    success = False
    attempts = 0

    for attempt in range(1, 21):
        attempts = attempt

        try:
            response = requests.get(
                f"{APP_URL}/work",
                timeout=5,
            )

            if response.status_code < 400:
                success = True
                break

        except requests.RequestException:
            pass

        time.sleep(0.5)

    mttr_ms = (
        time.perf_counter()
        - recovery_started
    ) * 1000

    return {
        "success": success,
        "mttr_ms": mttr_ms,
        "verification_attempts": attempts,
        "description": description,
    }


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
    "ce",
    "ch",
    "cc",
    "cr",
    "confidence",
    "confidence_margin",
    "correct",
    "decision_latency_ms",
    "recovery_success",
    "mttr_ms",
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
        "recovery_success": int(
            recovery["success"]
        ),
        "mttr_ms": round(
            recovery["mttr_ms"],
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
            "cr": 0.89,
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
            "cr": 0.91,
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
            "cr": 0.40,
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

    selected = ranked[0]
    second = ranked[1]

    recovery = execute_recovery(
        selected["decision"]
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
            "cr": 0.93,
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
            "cr": 0.87,
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
            "cr": 0.40,
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

    selected = ranked[0]
    second = ranked[1]

    recovery = execute_recovery(
        selected["decision"]
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
            "cr": 0.90,
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
            "cr": 0.87,
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
            "cr": 0.78,
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
            "cr": 0.30,
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

    selected = ranked[0]
    second = ranked[1]

    recovery = execute_recovery(
        selected["decision"]
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
            "cr": 0.91,
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
            "cr": 0.90,
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
            "cr": 0.86,
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
            "cr": 0.30,
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

    selected = ranked[0]
    second = ranked[1]

    recovery = execute_recovery(
        selected["decision"]
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
            "cr": 0.92,
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
            "cr": 0.90,
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
            "cr": 0.84,
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
            "cr": 0.25,
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

    selected = ranked[0]
    second = ranked[1]

    recovery = execute_recovery(
        selected["decision"]
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
                                {},
                            ).get(
                                "decision",
                                "",
                            )
                        ),
                        "confidence": (
                            payload_data.get(
                                "selected_decision",
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
                        "mttr_ms": (
                            payload_data.get(
                                "recovery",
                                {},
                            ).get(
                                "mttr_ms",
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
        repetitions=repetitions,
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

import threading
import uuid


CAMPAIGN_FILE = DATA_FILE.parent / "campaign_results.csv"

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
    "expected_decision",
    "correct",
    "confidence",
    "confidence_margin",
    "decision_latency_ms",
    "recovery_success",
    "mttr_ms",
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
    repetitions,
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
        for severity in CAMPAIGN_LEVELS[scenario]:
            for repetition in range(
                1,
                repetitions + 1,
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
                "decision_latency_ms": (
                    result.get(
                        "decision_latency_ms",
                        0,
                    )
                ),
                "recovery_success": int(
                    recovery["success"]
                ),
                "mttr_ms": recovery["mttr_ms"],
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

    repetitions = int(
        payload.get(
            "repetitions",
            5,
        )
    )

    repetitions = max(
        1,
        min(repetitions, 20),
    )

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

    campaign_id = str(uuid.uuid4())

    total = sum(
        len(CAMPAIGN_LEVELS[scenario])
        * repetitions
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
            repetitions,
            selected_scenarios,
        ),
        daemon=True,
    )

    thread.start()

    return jsonify(
        status="started",
        campaign_id=campaign_id,
        repetitions=repetitions,
        scenarios=selected_scenarios,
        total_episodes=total,
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


# ============================================================
# General API routes
# ============================================================

@app.get("/health")
def health():
    return jsonify(
        status="ok",
        service="ECDM Engine",
        data_file=str(DATA_FILE),
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
        ]
    )


# This block must remain at the very end of the file.
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8100,
        threaded=True,
    )
