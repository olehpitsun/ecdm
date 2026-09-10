import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


# ============================================================
# Application configuration
# ============================================================

st.set_page_config(
    page_title="ECDM Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENGINE_URL = (
    os.getenv("ECDM_ENGINE_URL")
    or os.getenv("ENGINE_URL")
    or "http://engine:8100"
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("ECDM_DASHBOARD_TIMEOUT", "180")
)

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "CPU Resource Exhaustion": {
        "key": "cpu",
        "endpoint": "/run/cpu",
        "severities": [0.60, 0.70, 0.80, 0.90, 0.95],
        "severity_label": "CPU load coefficient",
        "format": lambda value: f"{value:.2f}",
    },
    "Memory Pressure": {
        "key": "memory",
        "endpoint": "/run/memory",
        "severities": [100, 150, 200, 250, 300],
        "severity_label": "Injected memory",
        "format": lambda value: f"{int(value)} MB",
    },
    "Application Latency": {
        "key": "latency",
        "endpoint": "/run/latency",
        "severities": [300, 500, 700, 900, 1200],
        "severity_label": "Injected latency",
        "format": lambda value: f"{int(value)} ms",
    },
    "Application Error Injection": {
        "key": "errors",
        "endpoint": "/run/errors",
        "severities": [0.10, 0.20, 0.30, 0.35, 0.40],
        "severity_label": "Injected error probability",
        "format": lambda value: f"{value:.2f}",
    },
    "Service Dependency Failure": {
        "key": "dependency",
        "endpoint": "/run/dependency",
        "severities": [1.0],
        "severity_label": "Dependency failure indicator",
        "format": lambda value: "Unavailable" if float(value) >= 1.0 else "Available",
    },
}

DEFAULT_REPETITIONS = {
    "cpu": 6,
    "memory": 5,
    "latency": 5,
    "errors": 5,
    "dependency": 5,
}


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #f4f6f9;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    .ecdm-small {
        color: #667085;
        font-size: 0.90rem;
    }
    .ecdm-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        margin: 0.35rem 0 0.9rem 0;
        background: rgba(255,255,255,0.55);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API helpers
# ============================================================

class EngineAPIError(RuntimeError):
    pass


def _decode_json(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:500]
        raise EngineAPIError(
            f"Engine returned a non-JSON response (HTTP {response.status_code}): {preview}"
        ) from exc

    if not response.ok:
        message = payload.get("message") or payload.get("error") or str(payload)
        raise EngineAPIError(
            f"Engine request failed (HTTP {response.status_code}): {message}"
        )

    return payload


def api_get(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15,
) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{ENGINE_URL}{path}",
            params=params,
            timeout=timeout,
        )
        return _decode_json(response)
    except requests.RequestException as exc:
        raise EngineAPIError(f"Cannot reach ECDM Engine at {ENGINE_URL}: {exc}") from exc


def api_post(
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{ENGINE_URL}{path}",
            json=payload or {},
            timeout=timeout,
        )
        return _decode_json(response)
    except requests.RequestException as exc:
        raise EngineAPIError(f"Cannot reach ECDM Engine at {ENGINE_URL}: {exc}") from exc


# ============================================================
# Formatting helpers
# ============================================================

def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_ms(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f} ms"


def fmt_pct_from_fraction(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return f"{100.0 * number:.{digits}f}%"


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}%"


def yes_no(value: Any) -> str:
    if value is None:
        return "N/A"
    return "Yes" if bool(value) else "No"


def get_health() -> Optional[Dict[str, Any]]:
    try:
        return api_get("/health", timeout=5)
    except EngineAPIError:
        return None


def selected_decision_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = result.get("selected_decision") or {}
    return payload if isinstance(payload, dict) else {}


def recovery_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = result.get("recovery") or {}
    return payload if isinstance(payload, dict) else {}


def safety_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = result.get("safety") or {}
    return payload if isinstance(payload, dict) else {}


def observation_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = result.get("observation") or {}
    return payload if isinstance(payload, dict) else {}


def availability_payload(result: Dict[str, Any], key: str) -> Dict[str, Any]:
    payload = result.get(key) or {}
    return payload if isinstance(payload, dict) else {}


# ============================================================
# Result rendering
# ============================================================

def render_single_result(result: Dict[str, Any]) -> None:
    selected = selected_decision_payload(result)
    recovery = recovery_payload(result)
    safety = safety_payload(result)
    observation = observation_payload(result)
    pre = availability_payload(result, "pre_recovery")
    post = availability_payload(result, "post_recovery")

    selected_name = selected.get("decision", "N/A")
    expected = result.get("expected_decision", "N/A")
    confidence = as_float(selected.get("confidence"))
    decision_latency = result.get("decision_latency_ms")

    st.success("Experiment completed successfully.")

    top = st.columns(4)
    top[0].metric("Selected Decision", selected_name)
    top[1].metric(
        "Confidence",
        "N/A" if confidence is None else f"{confidence:.3f}",
    )
    top[2].metric("Reference Action", expected)
    top[3].metric(
        "Decision Computation Latency",
        fmt_ms(decision_latency, 3),
    )

    agreement = selected_name == expected if expected not in (None, "", "N/A") else None
    if agreement is True:
        st.info("Selected action agrees with the predefined reference policy.")
    elif agreement is False:
        st.warning(
            "Selected action does not agree with the predefined reference policy. "
            "Check the Safety Gate outcome before interpreting this as an executed recovery."
        )

    st.subheader("Runtime Observation")
    obs_cols = st.columns(5)
    obs_cols[0].metric("CPU Usage", fmt_pct(observation.get("cpu")))
    obs_cols[1].metric("Memory Usage", fmt_pct(observation.get("memory")))
    obs_cols[2].metric(
        "Allocated Memory",
        "N/A"
        if as_float(observation.get("allocated_memory_mb")) is None
        else f"{as_float(observation.get('allocated_memory_mb')):.2f} MB",
    )
    obs_cols[3].metric("Latency", fmt_ms(observation.get("latency_ms"), 2))
    obs_cols[4].metric("Error Rate", fmt_pct(observation.get("error_rate")))

    dep = as_float(observation.get("dependency_available"))
    if dep is not None:
        st.caption(f"Dependency availability signal: {dep:.3f}")

    st.subheader("Safety Gate")
    safety_cols = st.columns(4)
    safety_cols[0].metric("Autonomous Execution", yes_no(safety.get("allowed")))
    safety_cols[1].metric("Status", str(safety.get("status", "N/A")))
    margin = as_float(safety.get("confidence_margin"))
    safety_cols[2].metric(
        "Confidence Margin",
        "N/A" if margin is None else f"{margin:.3f}",
    )
    safety_cols[3].metric(
        "Executed Decision",
        recovery.get("executed_decision", "N/A"),
    )

    reason = safety.get("reason") or recovery.get("safety_reason")
    if reason:
        st.warning(f"Safety Gate: {reason}")
    elif safety.get("allowed"):
        st.success("Safety Gate admitted the decision for autonomous handling.")

    st.subheader("Recovery and Verification")
    recovery_attempted = bool(recovery.get("recovery_attempted", False))
    recovery_success = recovery.get("success")
    recovery_latency = recovery.get("recovery_verification_latency_ms")

    recovery_cols = st.columns(4)
    recovery_cols[0].metric("Active Recovery Attempted", yes_no(recovery_attempted))
    recovery_cols[1].metric(
        "Recovery Success",
        "N/A" if not recovery_attempted else yes_no(recovery_success),
    )
    recovery_cols[2].metric(
        "Recovery Verification Latency",
        fmt_ms(recovery_latency, 2) if recovery_attempted else "N/A",
    )
    recovery_cols[3].metric(
        "Verification Attempts",
        str(recovery.get("verification_attempts", 0)) if recovery_attempted else "N/A",
    )

    description = recovery.get("description")
    if description:
        st.caption(description)

    st.subheader("Availability Indicator")
    pre_availability = as_float(pre.get("availability"))
    post_availability = as_float(post.get("availability"))
    availability_gain = None
    if pre_availability is not None and post_availability is not None:
        availability_gain = post_availability - pre_availability

    av_cols = st.columns(3)
    av_cols[0].metric("Pre-recovery Availability", fmt_pct_from_fraction(pre_availability))
    av_cols[1].metric("Post-recovery Availability", fmt_pct_from_fraction(post_availability))
    av_cols[2].metric(
        "Availability Change",
        "N/A" if availability_gain is None else f"{100.0 * availability_gain:+.2f} pp",
    )

    ranked = result.get("ranked_decisions") or []
    if isinstance(ranked, list) and ranked:
        st.subheader("HDCM Candidate Ranking")
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(ranked, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "Rank": index,
                    "Decision": item.get("decision", ""),
                    "C_E": as_float(item.get("ce"), 0.0),
                    "C_H": as_float(item.get("ch"), 0.0),
                    "C_C": as_float(item.get("cc"), 0.0),
                    "C_R": as_float(item.get("cr"), 0.0),
                    "Confidence": as_float(item.get("confidence"), 0.0),
                    "Aggregation": item.get("aggregation_mode", ""),
                }
            )
        frame = pd.DataFrame(rows)
        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "C_E": st.column_config.NumberColumn(format="%.3f"),
                "C_H": st.column_config.NumberColumn(format="%.3f"),
                "C_C": st.column_config.NumberColumn(format="%.3f"),
                "C_R": st.column_config.NumberColumn(format="%.3f"),
                "Confidence": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    explanation = result.get("explanation")
    if explanation:
        st.subheader("Structured Decision Justification")
        st.write(explanation)

    with st.expander("Raw engine response"):
        st.json(result)


# ============================================================
# Sidebar: engine status and experiment configuration
# ============================================================

health = get_health()

with st.sidebar:
    if health:
        st.success("Engine: ok")
    else:
        st.error("Engine: unavailable")

    with st.expander("Engine connection"):
        st.code(ENGINE_URL)
        if health:
            st.json(
                {
                    "service": health.get("service"),
                    "aggregation": health.get("hdcm_aggregation_mode"),
                    "weights": health.get("hdcm_weights"),
                    "min_confidence": health.get("minimum_autonomous_confidence"),
                    "min_margin": health.get("minimum_confidence_margin"),
                    "execution_mode": health.get("action_execution_mode"),
                }
            )

    st.markdown("### Experiment Configuration")

    scenario_name = st.selectbox(
        "Failure Scenario",
        list(SCENARIOS.keys()),
        index=1,
    )
    scenario = SCENARIOS[scenario_name]

    severity = st.selectbox(
        scenario["severity_label"],
        scenario["severities"],
        index=len(scenario["severities"]) - 1,
        format_func=scenario["format"],
    )

    st.markdown("**Current experiment**")
    st.markdown(
        "- fault injection;\n"
        "- Prometheus monitoring;\n"
        "- evidence-oriented scoring;\n"
        "- HDCM evaluation;\n"
        "- Safety Gate;\n"
        "- recovery execution;\n"
        "- verification;\n"
        "- CSV logging."
    )

    run_clicked = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
        disabled=health is None,
    )

    if st.button("Clear Dashboard", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_error", None)
        st.session_state.pop("campaign_id", None)
        st.rerun()


if run_clicked:
    st.session_state.pop("last_error", None)
    try:
        with st.spinner(
            f"Running {scenario_name} at severity {scenario['format'](severity)}..."
        ):
            result = api_post(
                scenario["endpoint"],
                {"severity": severity},
            )
        st.session_state["last_result"] = result
    except EngineAPIError as exc:
        st.session_state["last_error"] = str(exc)


# ============================================================
# Main page
# ============================================================

st.title("ECDM Research Platform")
st.caption(
    "Interactive evaluation of evidence-centered autonomous recovery in self-healing software systems."
)

single_tab, campaign_tab, results_tab, history_tab = st.tabs(
    [
        "Single Experiment",
        "Confirmatory Campaign",
        "Results",
        "Recovery History",
    ]
)


with single_tab:
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    result = st.session_state.get("last_result")
    if result:
        render_single_result(result)
    else:
        st.info(
            "Choose a failure scenario and severity in the sidebar, then click **Run Experiment**."
        )
        st.markdown(
            "The dashboard reports **Decision Computation Latency** separately from "
            "**Recovery Verification Latency**."
        )


with campaign_tab:
    st.subheader("110-Episode Confirmatory Campaign")
    st.caption(
        "The default configuration reproduces the confirmatory design: "
        "CPU 30 episodes, Memory 25, Latency 25, Errors 25, Dependency 5."
    )

    selected_campaign_scenarios: List[str] = []
    repetitions_by_scenario: Dict[str, int] = {}

    campaign_columns = st.columns(5)
    campaign_items = [
        ("cpu", "CPU", 6),
        ("memory", "Memory", 5),
        ("latency", "Latency", 5),
        ("errors", "Errors", 5),
        ("dependency", "Dependency", 5),
    ]

    for column, (key, label, default_reps) in zip(campaign_columns, campaign_items):
        with column:
            enabled = st.checkbox(label, value=True, key=f"campaign_enabled_{key}")
            repetitions = st.number_input(
                "Repetitions / severity",
                min_value=1,
                max_value=20,
                value=default_reps,
                step=1,
                key=f"campaign_reps_{key}",
            )
            if enabled:
                selected_campaign_scenarios.append(key)
                repetitions_by_scenario[key] = int(repetitions)

    reset_history = st.checkbox(
        "Reset historical recovery confidence before campaign",
        value=True,
    )

    campaign_start_col, campaign_refresh_col = st.columns([1, 1])
    with campaign_start_col:
        if st.button(
            "Start Campaign",
            type="primary",
            use_container_width=True,
            disabled=health is None or not selected_campaign_scenarios,
        ):
            try:
                payload = {
                    "scenarios": selected_campaign_scenarios,
                    "repetitions_by_scenario": repetitions_by_scenario,
                    "reset_history": reset_history,
                }
                started = api_post("/campaign/start", payload, timeout=20)
                st.session_state["campaign_id"] = started.get("campaign_id")
                st.session_state["campaign_start_response"] = started
                st.success(
                    f"Campaign started: {started.get('total_episodes', 'N/A')} episodes."
                )
            except EngineAPIError as exc:
                st.error(str(exc))

    with campaign_refresh_col:
        refresh_campaign = st.button("Refresh Campaign Status", use_container_width=True)

    campaign_id = st.session_state.get("campaign_id")
    if campaign_id or refresh_campaign:
        try:
            status = api_get("/campaign/status", timeout=10)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", status.get("status", "N/A"))
            c2.metric("Total Episodes", status.get("total_episodes", 0))
            c3.metric("Completed", status.get("completed_episodes", 0))
            c4.metric("Failed", status.get("failed_episodes", 0))

            total = int(status.get("total_episodes") or 0)
            completed = int(status.get("completed_episodes") or 0)
            failed = int(status.get("failed_episodes") or 0)
            if total > 0:
                progress = min(1.0, (completed + failed) / total)
                st.progress(progress)

            current_scenario = status.get("current_scenario")
            current_severity = status.get("current_severity")
            if current_scenario:
                st.caption(
                    f"Current scenario: {current_scenario}; severity: {current_severity}"
                )
            if status.get("last_error"):
                st.warning(f"Last campaign error: {status['last_error']}")
        except EngineAPIError as exc:
            st.error(str(exc))

    if campaign_id:
        try:
            campaign_results = api_get(
                "/campaign/results",
                params={"campaign_id": campaign_id},
                timeout=20,
            )
            rows = campaign_results.get("results") or []
            if rows:
                st.markdown("#### Current campaign data")
                campaign_df = pd.DataFrame(rows)
                preferred_columns = [
                    "scenario",
                    "severity",
                    "expected_decision",
                    "selected_decision",
                    "executed_decision",
                    "correct",
                    "confidence",
                    "confidence_margin",
                    "decision_latency_ms",
                    "recovery_attempted",
                    "recovery_success",
                    "operator_escalation",
                    "recovery_verification_latency_ms",
                    "pre_availability",
                    "post_availability",
                    "availability_gain",
                ]
                visible = [column for column in preferred_columns if column in campaign_df.columns]
                st.dataframe(
                    campaign_df[visible],
                    use_container_width=True,
                    hide_index=True,
                )
        except EngineAPIError as exc:
            st.warning(str(exc))


with results_tab:
    st.subheader("Recorded Experiment Results")
    if st.button("Reload Results", key="reload_results") or True:
        try:
            result_payload = api_get("/results", timeout=20)
            rows = result_payload.get("results") or []
            if not rows:
                st.info("No experiment results are currently available.")
            else:
                df = pd.DataFrame(rows)
                st.caption(f"Rows available: {len(df)}")

                preferred = [
                    "timestamp",
                    "scenario",
                    "severity",
                    "expected_decision",
                    "selected_decision",
                    "executed_decision",
                    "correct",
                    "ce",
                    "ch",
                    "cc",
                    "cr",
                    "confidence",
                    "confidence_margin",
                    "decision_latency_ms",
                    "recovery_attempted",
                    "recovery_success",
                    "operator_escalation",
                    "safety_status",
                    "recovery_verification_latency_ms",
                    "pre_availability",
                    "post_availability",
                    "availability_gain",
                ]
                visible = [column for column in preferred if column in df.columns]
                st.dataframe(
                    df[visible].tail(250),
                    use_container_width=True,
                    hide_index=True,
                )
        except EngineAPIError as exc:
            st.error(str(exc))


with history_tab:
    st.subheader("Historical Recovery Confidence")
    history_col, reset_col = st.columns([2, 1])

    try:
        history_payload = api_get("/history", timeout=10)
        prior_alpha = history_payload.get("prior_alpha")
        prior_beta = history_payload.get("prior_beta")
        history = history_payload.get("history") or {}

        with history_col:
            st.caption(
                f"Beta prior: alpha={prior_alpha}, beta={prior_beta}. "
                "With no previous record, the initial recovery confidence is 0.5."
            )
            if history:
                if isinstance(history, dict):
                    rows = []
                    for action, values in history.items():
                        if isinstance(values, dict):
                            row = {"Action": action, **values}
                        else:
                            row = {"Action": action, "Value": values}
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.json(history)
            else:
                st.info("Recovery history is empty; actions use the neutral cold-start prior.")

        with reset_col:
            if st.button("Reset Recovery History", use_container_width=True):
                try:
                    reset_payload = api_post("/history/reset", {}, timeout=10)
                    st.success(
                        "Recovery history reset. Initial C_R = "
                        f"{reset_payload.get('initial_recovery_confidence', 0.5):.3f}."
                    )
                    time.sleep(0.4)
                    st.rerun()
                except EngineAPIError as exc:
                    st.error(str(exc))

    except EngineAPIError as exc:
        st.error(str(exc))


st.divider()
st.caption(
    "ECDM prototype dashboard — decision computation and recovery verification are reported as separate metrics."
)
