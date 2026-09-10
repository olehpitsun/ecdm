from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path(
    "/data/campaign_results_confirmatory.csv"
)

OUTPUT_DIR = Path(
    "/data/article_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SCENARIO_NAMES = {
    "cpu": "CPU Resource Exhaustion",
    "memory": "Memory Pressure",
    "dependency": "Service Dependency Failure",
    "latency": "Application Latency",
    "errors": "Application Error Injection",
}


SEVERITY_LABELS = {
    "cpu": "CPU load coefficient",
    "memory": "Allocated memory (MB)",
    "dependency": "Dependency failure state",
    "latency": "Injected latency (ms)",
    "errors": "Error probability",
}


def confidence_interval_95(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return 0.0

    standard_error = (
        values.std(ddof=1)
        / math.sqrt(len(values))
    )

    return 1.96 * standard_error


def prepare_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    numeric_columns = [
        "severity",
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

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["Scenario"] = df["scenario"].map(
        SCENARIO_NAMES
    ).fillna(df["scenario"])

    return df


def create_overall_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario, group in df.groupby(
        "scenario",
        sort=False,
    ):
        rows.append(
            {
                "Scenario": SCENARIO_NAMES.get(
                    scenario,
                    scenario,
                ),
                "Runs": len(group),
                "Decision Accuracy": (
                    group["correct"].mean()
                ),
                "Recovery Success Rate": (
                    group["recovery_success"].mean()
                ),
                "Mean Confidence": (
                    group["confidence"].mean()
                ),
                "Confidence SD": (
                    group["confidence"].std(ddof=1)
                ),
                "Confidence 95% CI": (
                    confidence_interval_95(
                        group["confidence"]
                    )
                ),
                "Mean Confidence Margin": (
                    group[
                        "confidence_margin"
                    ].mean()
                ),
                "Mean Decision Latency (ms)": (
                    group[
                        "decision_latency_ms"
                    ].mean()
                ),
                "Mean MTTR (ms)": (
                    group["mttr_ms"].mean()
                ),
                "MTTR SD (ms)": (
                    group["mttr_ms"].std(ddof=1)
                ),
                "Mean Availability Before": (
                    group[
                        "pre_availability"
                    ].mean()
                ),
                "Mean Availability After": (
                    group[
                        "post_availability"
                    ].mean()
                ),
                "Mean Availability Gain": (
                    group[
                        "availability_gain"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "Table_Overall_Results.csv",
        index=False,
    )

    return summary


def create_severity_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    severity_summary = (
        df.groupby(
            [
                "scenario",
                "Scenario",
                "severity",
            ],
            as_index=False,
        )
        .agg(
            Runs=("correct", "size"),
            Decision_Accuracy=(
                "correct",
                "mean",
            ),
            Recovery_Success_Rate=(
                "recovery_success",
                "mean",
            ),
            Mean_Confidence=(
                "confidence",
                "mean",
            ),
            Confidence_SD=(
                "confidence",
                "std",
            ),
            Mean_Confidence_Margin=(
                "confidence_margin",
                "mean",
            ),
            Mean_MTTR_ms=(
                "mttr_ms",
                "mean",
            ),
            MTTR_SD_ms=(
                "mttr_ms",
                "std",
            ),
            Mean_Availability_Before=(
                "pre_availability",
                "mean",
            ),
            Mean_Availability_After=(
                "post_availability",
                "mean",
            ),
            Mean_Availability_Gain=(
                "availability_gain",
                "mean",
            ),
        )
    )

    severity_summary.to_csv(
        OUTPUT_DIR
        / "Table_Results_by_Severity.csv",
        index=False,
    )

    return severity_summary


def create_confidence_figures(
    severity_summary: pd.DataFrame,
) -> None:
    for scenario in [
        "cpu",
        "memory",
        "latency",
        "errors",
    ]:
        subset = severity_summary[
            severity_summary["scenario"]
            == scenario
        ].sort_values("severity")

        if subset.empty:
            continue

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.errorbar(
            subset["severity"],
            subset["Mean_Confidence"],
            yerr=subset[
                "Confidence_SD"
            ].fillna(0),
            marker="o",
            capsize=4,
        )

        plt.xlabel(
            SEVERITY_LABELS[scenario]
        )
        plt.ylabel(
            "Mean overall decision confidence"
        )
        plt.ylim(0, 1.05)
        plt.grid(
            axis="y",
            alpha=0.25,
        )
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / (
                "Figure11_Confidence_"
                f"{scenario}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.savefig(
            OUTPUT_DIR
            / (
                "Figure11_Confidence_"
                f"{scenario}.svg"
            ),
            bbox_inches="tight",
        )

        plt.close()


def create_mttr_figures(
    severity_summary: pd.DataFrame,
) -> None:
    for scenario in [
        "cpu",
        "memory",
        "latency",
        "errors",
    ]:
        subset = severity_summary[
            severity_summary["scenario"]
            == scenario
        ].sort_values("severity")

        if subset.empty:
            continue

        plt.figure(
            figsize=(7.2, 4.8)
        )

        plt.errorbar(
            subset["severity"],
            subset["Mean_MTTR_ms"],
            yerr=subset[
                "MTTR_SD_ms"
            ].fillna(0),
            marker="o",
            capsize=4,
        )

        plt.xlabel(
            SEVERITY_LABELS[scenario]
        )
        plt.ylabel(
            "Mean Time to Recovery (ms)"
        )
        plt.grid(
            axis="y",
            alpha=0.25,
        )
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / (
                "Figure12_MTTR_"
                f"{scenario}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.savefig(
            OUTPUT_DIR
            / (
                "Figure12_MTTR_"
                f"{scenario}.svg"
            ),
            bbox_inches="tight",
        )

        plt.close()


def create_accuracy_success_figure(
    summary: pd.DataFrame,
) -> None:
    chart_data = summary.set_index(
        "Scenario"
    )[
        [
            "Decision Accuracy",
            "Recovery Success Rate",
        ]
    ]

    plt.figure(
        figsize=(10, 5.5)
    )

    axis = chart_data.plot(
        kind="bar",
        ax=plt.gca(),
    )

    axis.set_xlabel(
        "Failure scenario"
    )
    axis.set_ylabel(
        "Rate"
    )
    axis.set_ylim(
        0,
        1.05,
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        loc="lower right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure13_Accuracy_and_Recovery.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT_DIR
        / "Figure13_Accuracy_and_Recovery.svg",
        bbox_inches="tight",
    )

    plt.close()


def create_decision_distribution(
    df: pd.DataFrame,
) -> None:
    distribution = pd.crosstab(
        df["Scenario"],
        df["selected_decision"],
    )

    distribution.to_csv(
        OUTPUT_DIR
        / "Table_Decision_Distribution.csv"
    )

    plt.figure(
        figsize=(10, 5.5)
    )

    axis = distribution.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
    )

    axis.set_xlabel(
        "Failure scenario"
    )
    axis.set_ylabel(
        "Number of selected decisions"
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        title="Recovery action",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure14_Decision_Distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT_DIR
        / "Figure14_Decision_Distribution.svg",
        bbox_inches="tight",
    )

    plt.close()


def create_availability_figure(
    summary: pd.DataFrame,
) -> None:
    chart_data = summary.set_index(
        "Scenario"
    )[
        [
            "Mean Availability Before",
            "Mean Availability After",
        ]
    ]

    plt.figure(
        figsize=(10, 5.5)
    )

    axis = chart_data.plot(
        kind="bar",
        ax=plt.gca(),
    )

    axis.set_xlabel(
        "Failure scenario"
    )
    axis.set_ylabel(
        "Mean service availability"
    )
    axis.set_ylim(
        0,
        1.05,
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "Figure15_Availability_Recovery.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT_DIR
        / "Figure15_Availability_Recovery.svg",
        bbox_inches="tight",
    )

    plt.close()


def create_manuscript_table(
    summary: pd.DataFrame,
) -> None:
    manuscript = summary.copy()

    percentage_columns = [
        "Decision Accuracy",
        "Recovery Success Rate",
        "Mean Availability Before",
        "Mean Availability After",
        "Mean Availability Gain",
    ]

    for column in percentage_columns:
        manuscript[column] = (
            manuscript[column]
            * 100
        ).round(2)

    round_columns = [
        "Mean Confidence",
        "Confidence SD",
        "Confidence 95% CI",
        "Mean Confidence Margin",
        "Mean Decision Latency (ms)",
        "Mean MTTR (ms)",
        "MTTR SD (ms)",
    ]

    for column in round_columns:
        manuscript[column] = (
            manuscript[column]
            .round(3)
        )

    selected_columns = [
        "Scenario",
        "Runs",
        "Decision Accuracy",
        "Recovery Success Rate",
        "Mean Confidence",
        "Mean Confidence Margin",
        "Mean Decision Latency (ms)",
        "Mean MTTR (ms)",
        "Mean Availability Gain",
    ]

    manuscript = manuscript[
        selected_columns
    ]

    manuscript = manuscript.rename(
        columns={
            "Decision Accuracy":
                "Decision Accuracy (%)",
            "Recovery Success Rate":
                "Recovery Success (%)",
            "Mean Availability Gain":
                "Availability Gain (pp)",
        }
    )

    manuscript.to_csv(
        OUTPUT_DIR
        / "Table_Manuscript_Ready.csv",
        index=False,
    )

    markdown = manuscript.to_markdown(
        index=False,
    )

    (
        OUTPUT_DIR
        / "Table_Manuscript_Ready.md"
    ).write_text(
        markdown,
        encoding="utf-8",
    )


def create_results_text(
    df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    total_runs = len(df)

    overall_accuracy = (
        df["correct"].mean()
    )

    overall_success = (
        df["recovery_success"].mean()
    )

    overall_confidence = (
        df["confidence"].mean()
    )

    overall_mttr = (
        df["mttr_ms"].mean()
    )

    overall_gain = (
        df["availability_gain"].mean()
    )

    best_accuracy_row = summary.loc[
        summary[
            "Decision Accuracy"
        ].idxmax()
    ]

    highest_confidence_row = summary.loc[
        summary[
            "Mean Confidence"
        ].idxmax()
    ]

    lowest_mttr_row = summary.loc[
        summary[
            "Mean MTTR (ms)"
        ].idxmin()
    ]

    text = f"""Experimental Results

The experimental campaign comprised {total_runs} autonomous recovery episodes covering CPU resource exhaustion, memory pressure, service dependency failure, application latency, and application error injection. Each non-binary fault type was evaluated at multiple severity levels, and every configuration was repeatedly executed to reduce the influence of stochastic variation in workload generation and monitoring.

Across all experimental episodes, the proposed ECDM framework achieved an overall decision accuracy of {overall_accuracy * 100:.2f}% and a recovery success rate of {overall_success * 100:.2f}%. The mean overall decision confidence was {overall_confidence:.3f}, while the mean time to recovery was {overall_mttr:.3f} ms. On average, the executed recovery actions increased service availability by {overall_gain * 100:.2f} percentage points.

The highest decision accuracy was observed for the {best_accuracy_row['Scenario']} scenario, with an accuracy of {best_accuracy_row['Decision Accuracy'] * 100:.2f}%. The highest mean confidence was obtained for {highest_confidence_row['Scenario']} ({highest_confidence_row['Mean Confidence']:.3f}). The lowest mean recovery time was measured for {lowest_mttr_row['Scenario']} ({lowest_mttr_row['Mean MTTR (ms)']:.3f} ms).

The severity-level analysis demonstrates how the confidence assigned by the HDCM changes as the intensity of the injected fault increases. For CPU exhaustion, memory pressure, latency, and application error scenarios, the results are reported separately for every tested severity level. This analysis makes it possible to determine whether stronger runtime symptoms lead to clearer evidence, stronger hypothesis support, and a larger confidence margin between the selected recovery action and the competing alternatives.

The availability measurements confirm that the framework does not only produce a recovery recommendation but also verifies the operational outcome of the selected action. The comparison of availability before and after recovery provides an outcome-based measure of practical recovery effectiveness. The recorded MTTR values additionally quantify the time required from the beginning of recovery execution until a successful service response is observed.
"""

    (
        OUTPUT_DIR
        / "Results_Section_Draft.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:
    df = prepare_data()

    summary = create_overall_summary(
        df
    )

    severity_summary = (
        create_severity_summary(df)
    )

    create_confidence_figures(
        severity_summary
    )

    create_mttr_figures(
        severity_summary
    )

    create_accuracy_success_figure(
        summary
    )

    create_decision_distribution(
        df
    )

    create_availability_figure(
        summary
    )

    create_manuscript_table(
        summary
    )

    create_results_text(
        df,
        summary,
    )

    print(
        f"Analyzed episodes: {len(df)}"
    )

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
