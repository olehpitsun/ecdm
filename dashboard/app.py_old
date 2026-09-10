import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


ENGINE_URL = "http://ecdm-engine:8100"


st.set_page_config(
    page_title="ECDM Research Platform",
    page_icon="🧠",
    layout="wide",
)


def engine_get(path: str) -> dict[str, Any]:
    response = requests.get(
        f"{ENGINE_URL}{path}",
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def engine_post(
    path: str,
    timeout: int = 90,
) -> dict[str, Any]:
    response = requests.post(
        f"{ENGINE_URL}{path}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def draw_evidence_graph(result: dict[str, Any]):
    observation = result["observation"]
    selected = result["selected_decision"]

    nodes = {
        "CPU Usage": (0, 3),
        "Latency": (2, 3),
        "Memory": (4, 3),
        "Error Rate": (6, 3),

        "High CPU Evidence": (1, 2),
        "Performance Degradation": (3, 2),
        "Normal Memory Evidence": (5, 2),

        "CPU Resource Exhaustion": (3, 1),
        selected["decision"]: (3, 0),
    }

    edges = [
        ("CPU Usage", "High CPU Evidence"),
        ("Latency", "Performance Degradation"),
        ("Memory", "Normal Memory Evidence"),
        ("Error Rate", "Performance Degradation"),

        ("High CPU Evidence", "CPU Resource Exhaustion"),
        ("Performance Degradation", "CPU Resource Exhaustion"),
        ("Normal Memory Evidence", "CPU Resource Exhaustion"),

        ("CPU Resource Exhaustion", selected["decision"]),
    ]

    figure = go.Figure()

    for start, end in edges:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]

        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=2,
                    color="black",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    labels = []
    hover = []

    for name, (x, y) in nodes.items():
        labels.append(name)

        if name == "CPU Usage":
            hover.append(
                f"CPU: {observation['cpu']:.2f}%"
            )
        elif name == "Latency":
            hover.append(
                f"Latency: {observation['latency_ms']:.2f} ms"
            )
        elif name == "Memory":
            hover.append(
                f"Memory: {observation['memory']:.2f}%"
            )
        elif name == "Error Rate":
            hover.append(
                f"Error rate: {observation['error_rate']:.2f}%"
            )
        else:
            hover.append(name)

    figure.add_trace(
        go.Scatter(
            x=[nodes[name][0] for name in nodes],
            y=[nodes[name][1] for name in nodes],
            mode="markers+text",
            text=labels,
            textposition="bottom center",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=30,
                color="white",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
            showlegend=False,
        )
    )

    figure.update_layout(
        title="Evidence Graph",
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=40,
        ),
        xaxis=dict(
            visible=False,
            range=[-1, 7],
        ),
        yaxis=dict(
            visible=False,
            range=[-0.5, 3.7],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return figure


def draw_radar(result: dict[str, Any]):
    candidates = result["ranked_decisions"]

    categories = [
        "Evidence Confidence",
        "Hypothesis Confidence",
        "Context Confidence",
        "Recovery Confidence",
    ]

    figure = go.Figure()

    for candidate in candidates:
        values = [
            candidate["ce"],
            candidate["ch"],
            candidate["cc"],
            candidate["cr"],
        ]

        figure.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=candidate["decision"],
            )
        )

    figure.update_layout(
        title="HDCM Confidence Profiles",
        height=500,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
        ),
    )

    return figure


def draw_ranking(result: dict[str, Any]):
    ranked = pd.DataFrame(
        result["ranked_decisions"]
    )

    ranked = ranked.sort_values(
        "confidence",
        ascending=True,
    )

    figure = go.Figure(
        go.Bar(
            x=ranked["confidence"],
            y=ranked["decision"],
            orientation="h",
            text=ranked["confidence"].round(3),
            textposition="outside",
        )
    )

    figure.update_layout(
        title="Candidate Decision Ranking",
        height=380,
        xaxis=dict(
            title="Overall Decision Confidence",
            range=[0, 1],
        ),
        yaxis=dict(
            title="Recovery Action",
        ),
        margin=dict(
            l=20,
            r=40,
            t=60,
            b=40,
        ),
    )

    return figure


def draw_timeline(result: dict[str, Any]):
    recovery = result["recovery"]

    stages = [
        "Fault Injection",
        "Observation Collection",
        "Evidence Construction",
        "Hypothesis Generation",
        "HDCM Evaluation",
        "Decision Selection",
        "Recovery Execution",
        "Verification",
        "Knowledge Update",
    ]

    durations = [
        1.0,
        1.2,
        0.4,
        0.3,
        0.2,
        0.1,
        max(
            recovery["mttr_ms"] / 1000,
            0.1,
        ),
        0.4,
        0.2,
    ]

    starts = []
    current = 0.0

    for duration in durations:
        starts.append(current)
        current += duration

    figure = go.Figure()

    for stage, start, duration in zip(
        stages,
        starts,
        durations,
    ):
        figure.add_trace(
            go.Bar(
                x=[duration],
                y=["ECDM Recovery Cycle"],
                base=[start],
                orientation="h",
                name=stage,
                hovertemplate=(
                    f"{stage}<br>"
                    f"Duration: {duration:.3f} s"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="Evidence-Centered Recovery Timeline",
        barmode="stack",
        height=320,
        xaxis_title="Relative Time (s)",
        legend=dict(
            orientation="h",
            y=-0.3,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=90,
        ),
    )

    return figure


def results_table():
    payload = engine_get("/results")

    if payload.get("status") == "empty":
        return pd.DataFrame()

    return pd.DataFrame(
        payload.get("results", [])
    )


st.title("ECDM Research Platform")

st.caption(
    "Interactive evaluation of evidence-centered "
    "autonomous recovery in self-healing software systems."
)

try:
    health = engine_get("/health")
    st.sidebar.success(
        f"Engine: {health['status']}"
    )
except Exception as error:
    st.sidebar.error(
        f"ECDM Engine unavailable: {error}"
    )
    st.stop()


st.sidebar.header("Experiment Configuration")

scenario = st.sidebar.selectbox(
    "Failure Scenario",
    [
        "CPU Resource Exhaustion",
        "Memory Pressure",
        "Service Dependency Failure",
        "Application Latency",
        "Application Error Injection",
    ],
)

st.sidebar.markdown(
    """
    **Current experiment**

    - fault injection;
    - Prometheus monitoring;
    - Evidence Graph;
    - HDCM evaluation;
    - recovery execution;
    - verification;
    - CSV logging.
    """
)

run_experiment = st.sidebar.button(
    "Run Experiment",
    type="primary",
    use_container_width=True,
)


if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


if run_experiment:
    with st.spinner(
        f"Running {scenario} experiment..."
    ):
        try:
            endpoint = {
                "CPU Resource Exhaustion": "/run/cpu",
                "Memory Pressure": "/run/memory",
                "Service Dependency Failure": "/run/dependency",
                "Application Latency": "/run/latency",
                "Application Error Injection": "/run/errors",
            }[scenario]

            result = engine_post(
                endpoint,
                timeout=120,
            )

            st.session_state.latest_result = result

            st.success(
                "Experiment completed successfully."
            )

        except requests.RequestException as error:
            st.error(
                f"Experiment failed: {error}"
            )


result = st.session_state.latest_result


if result is None:
    st.info(
        "Select the scenario and press "
        "'Run Experiment' to start."
    )

    history = results_table()

    if not history.empty:
        st.subheader("Previous Experiments")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

    st.stop()


selected = result["selected_decision"]
recovery = result["recovery"]
observation = result["observation"]
pre = result["pre_recovery"]
post = result["post_recovery"]


metric_columns = st.columns(6)

metric_columns[0].metric(
    "Selected Decision",
    selected["decision"],
)

metric_columns[1].metric(
    "Confidence",
    f"{selected['confidence']:.3f}",
)

metric_columns[2].metric(
    "CPU Usage",
    f"{observation['cpu']:.2f}%",
)

metric_columns[3].metric(
    "Latency",
    f"{observation['latency_ms']:.2f} ms",
)

metric_columns[4].metric(
    "MTTR",
    f"{recovery['mttr_ms']:.2f} ms",
)

metric_columns[5].metric(
    "Recovery",
    "Success" if recovery["success"] else "Failed",
)


tabs = st.tabs(
    [
        "Decision Overview",
        "Evidence Graph",
        "HDCM Analysis",
        "Recovery Timeline",
        "Experimental History",
    ]
)


with tabs[0]:
    st.subheader(
        "Automatic Decision Explanation"
    )

    st.success(
        result["explanation"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Observation")

        observation_df = pd.DataFrame(
            [
                {
                    "Observation": "CPU Usage",
                    "Value": f"{observation['cpu']:.4f}%",
                },
                {
                    "Observation": "Memory Usage",
                    "Value": f"{observation['memory']:.4f}%",
                },
        {
                "Observation": "Injected Memory",
            "Value": (
                f"{observation.get('allocated_memory_mb', 0):.4f} MB"
                ),
        },
                {
                    "Observation": "Latency",
                    "Value": f"{observation['latency_ms']:.4f} ms",
                },
                {
                    "Observation": "Error Rate",
                    "Value": f"{observation['error_rate']:.4f}%",
                },
                {
                    "Observation": "Dependency Availability",
                    "Value": (
                        f"{observation.get('dependency_available', 0):.4f}"
                    ),
                },
            ]
        )

        st.dataframe(
            observation_df,
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("#### Recovery Result")

        recovery_df = pd.DataFrame(
            [
                {
                    "Metric": "Recovery Success",
                    "Value": recovery["success"],
                },
                {
                    "Metric": "MTTR",
                    "Value": f"{recovery['mttr_ms']:.4f} ms",
                },
                {
                    "Metric": "Verification Attempts",
                    "Value": recovery["verification_attempts"],
                },
                {
                    "Metric": "Availability Before",
                    "Value": f"{pre['availability']:.4f}",
                },
                {
                    "Metric": "Availability After",
                    "Value": f"{post['availability']:.4f}",
                },
            ]
        )

        st.dataframe(
            recovery_df,
            use_container_width=True,
            hide_index=True,
        )

    st.plotly_chart(
        draw_ranking(result),
        use_container_width=True,
    )


with tabs[1]:
    st.plotly_chart(
        draw_evidence_graph(result),
        use_container_width=True,
    )

    st.markdown(
        """
        The graph preserves the complete reasoning path:

        **Runtime Observation → Structured Evidence → Failure Hypothesis → Recovery Decision**
        """
    )


with tabs[2]:
    st.plotly_chart(
        draw_radar(result),
        use_container_width=True,
    )

    ranked_df = pd.DataFrame(
        result["ranked_decisions"]
    )

    ranked_df = ranked_df.rename(
        columns={
            "decision": "Decision",
            "ce": "C_E",
            "ch": "C_H",
            "cc": "C_C",
            "cr": "C_R",
            "confidence": "C(d)",
        }
    )

    st.dataframe(
        ranked_df[
            [
                "Decision",
                "C_E",
                "C_H",
                "C_C",
                "C_R",
                "C(d)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


with tabs[3]:
    st.plotly_chart(
        draw_timeline(result),
        use_container_width=True,
    )

    st.markdown(
        f"""
        **Executed recovery:** {selected['decision']}

        **Result:** {"Successful" if recovery["success"] else "Failed"}

        **Verification attempts:** {recovery["verification_attempts"]}

        **MTTR:** {recovery["mttr_ms"]:.4f} ms
        """
    )


with tabs[4]:
    history = results_table()

    if history.empty:
        st.info(
            "No experiment history is available."
        )
    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Experimental Dataset",
            data=csv_data,
            file_name="experiment_results.csv",
            mime="text/csv",
        )
