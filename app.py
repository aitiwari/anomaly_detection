import os
import traceback
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================================================
# Configuration
# =========================================================

DEFAULT_FILE_CANDIDATES = [
    "20210708-NHS-111-MDS-time-series-to-March-2021.xlsx",
    "./20210708-NHS-111-MDS-time-series-to-March-2021.xlsx",
    "./data/20210708-NHS-111-MDS-time-series-to-March-2021.xlsx"
]


# =========================================================
# Helper functions
# =========================================================

def clean_column_name(col):
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("%", "percent")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("&", "and")
        .replace("-", "_")
        .replace(".", "")
    )


def find_existing_default_file():
    for path in DEFAULT_FILE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def build_columns_from_two_header_rows(df_raw, top_row_idx=4, bottom_row_idx=5):
    top_header = df_raw.iloc[top_row_idx]
    bottom_header = df_raw.iloc[bottom_row_idx]

    final_columns = []

    for top, bottom in zip(top_header, bottom_header):
        top_is_na = pd.isna(top)
        bottom_is_na = pd.isna(bottom)

        if top_is_na and bottom_is_na:
            final_columns.append(None)
        elif bottom_is_na:
            final_columns.append(str(top).strip())
        elif top_is_na:
            final_columns.append(str(bottom).strip())
        else:
            final_columns.append(f"{str(top).strip()}_{str(bottom).strip()}")

    return final_columns


def load_nhs_excel_raw_sheet(file_path):
    raw_df = pd.read_excel(file_path, sheet_name="Raw", header=None)

    final_columns = build_columns_from_two_header_rows(raw_df, 4, 5)

    df = raw_df.iloc[6:].copy()
    df.columns = final_columns
    df = df.reset_index(drop=True)

    df = df.loc[:, df.columns.notna()]
    df.columns = [clean_column_name(c) for c in df.columns]

    return df


def load_uploaded_or_default_file(uploaded_file):
    if uploaded_file is not None:
        file_path = uploaded_file.name
    else:
        file_path = find_existing_default_file()

    if not file_path:
        raise ValueError(
            "No NHS file found. Please upload the Excel file or keep "
            "20210708-NHS-111-MDS-time-series-to-March-2021.xlsx in the app folder."
        )

    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        df.columns = [clean_column_name(c) for c in df.columns]
        return df, file_path

    if file_path.lower().endswith(".xlsx") or file_path.lower().endswith(".xls"):
        try:
            df = load_nhs_excel_raw_sheet(file_path)
            return df, file_path
        except Exception:
            df = pd.read_excel(file_path)
            df.columns = [clean_column_name(c) for c in df.columns]
            return df, file_path

    raise ValueError("Unsupported file format. Please use Excel or CSV.")


def find_first_matching_column(columns, candidates):
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def prepare_nhs_dataset(df):
    columns = list(df.columns)

    region_col = find_first_matching_column(columns, ["region"])
    provider_col = find_first_matching_column(columns, ["provider"])
    period_col = find_first_matching_column(columns, ["period", "month", "date", "reporting_period"])
    code_col = find_first_matching_column(columns, ["code"])
    area_col = find_first_matching_column(columns, ["area"])
    population_col = find_first_matching_column(columns, ["43_population", "population", "4_3_population"])
    calls_offered_col = find_first_matching_column(
        columns,
        ["53_total_calls_offered", "5_3_total_calls_offered", "calls_offered", "total_calls_offered"]
    )
    calls_abandoned_col = find_first_matching_column(
        columns,
        ["56_no_of_abandoned_calls", "5_6_no_of_abandoned_calls", "calls_abandoned", "total_calls_abandoned"]
    )
    calls_answered_col = find_first_matching_column(
        columns,
        ["57_no_calls_answered", "5_7_no_calls_answered", "calls_answered", "total_calls_answered"]
    )

    required = [calls_offered_col, calls_abandoned_col, calls_answered_col]
    if any(col is None for col in required):
        raise ValueError(
            "Required columns not found. Expected calls offered, calls abandoned, and calls answered columns."
        )

    keep_cols = [
        c for c in [
            region_col,
            provider_col,
            period_col,
            code_col,
            area_col,
            population_col,
            calls_offered_col,
            calls_abandoned_col,
            calls_answered_col
        ] if c is not None
    ]

    clean_df = df[keep_cols].copy()

    rename_map = {}
    if region_col:
        rename_map[region_col] = "region"
    if provider_col:
        rename_map[provider_col] = "provider"
    if period_col:
        rename_map[period_col] = "period"
    if code_col:
        rename_map[code_col] = "code"
    if area_col:
        rename_map[area_col] = "area"
    if population_col:
        rename_map[population_col] = "population"

    rename_map[calls_offered_col] = "calls_offered"
    rename_map[calls_abandoned_col] = "calls_abandoned"
    rename_map[calls_answered_col] = "calls_answered"

    clean_df = clean_df.rename(columns=rename_map)

    numeric_cols = ["calls_offered", "calls_abandoned", "calls_answered"]
    if "population" in clean_df.columns:
        numeric_cols.append("population")

    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna(subset=["calls_offered", "calls_abandoned", "calls_answered"]).copy()
    clean_df = clean_df[clean_df["calls_offered"] > 0].copy()

    if "period" in clean_df.columns:
        clean_df["period"] = pd.to_datetime(clean_df["period"], errors="coerce")

    clean_df["calls_answered"] = np.minimum(clean_df["calls_answered"], clean_df["calls_offered"])
    clean_df["calls_abandoned"] = np.minimum(clean_df["calls_abandoned"], clean_df["calls_offered"])

    clean_df["unanswered_calls"] = clean_df["calls_offered"] - clean_df["calls_answered"]
    clean_df["answer_rate"] = np.where(
        clean_df["calls_offered"] > 0,
        clean_df["calls_answered"] / clean_df["calls_offered"],
        0
    )
    clean_df["abandon_rate"] = np.where(
        clean_df["calls_offered"] > 0,
        clean_df["calls_abandoned"] / clean_df["calls_offered"],
        0
    )

    if "population" in clean_df.columns:
        clean_df["calls_per_1000_population"] = np.where(
            clean_df["population"] > 0,
            (clean_df["calls_offered"] / clean_df["population"]) * 1000,
            0
        )
    else:
        clean_df["calls_per_1000_population"] = np.nan

    return clean_df


def create_synthetic_dataset(clean_df, anomaly_fraction=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    synthetic_df = clean_df.copy()

    synthetic_df["calls_offered"] = synthetic_df["calls_offered"] * rng.uniform(0.90, 1.10, len(synthetic_df))
    synthetic_df["calls_answered"] = synthetic_df["calls_answered"] * rng.uniform(0.90, 1.08, len(synthetic_df))
    synthetic_df["calls_abandoned"] = synthetic_df["calls_abandoned"] * rng.uniform(0.90, 1.20, len(synthetic_df))

    synthetic_df["calls_answered"] = np.minimum(synthetic_df["calls_answered"], synthetic_df["calls_offered"])
    synthetic_df["calls_abandoned"] = np.minimum(synthetic_df["calls_abandoned"], synthetic_df["calls_offered"])

    synthetic_df["unanswered_calls"] = synthetic_df["calls_offered"] - synthetic_df["calls_answered"]
    synthetic_df["answer_rate"] = np.where(
        synthetic_df["calls_offered"] > 0,
        synthetic_df["calls_answered"] / synthetic_df["calls_offered"],
        0
    )
    synthetic_df["abandon_rate"] = np.where(
        synthetic_df["calls_offered"] > 0,
        synthetic_df["calls_abandoned"] / synthetic_df["calls_offered"],
        0
    )

    if "population" in synthetic_df.columns:
        synthetic_df["calls_per_1000_population"] = np.where(
            synthetic_df["population"] > 0,
            (synthetic_df["calls_offered"] / synthetic_df["population"]) * 1000,
            0
        )

    synthetic_df["is_anomaly"] = 0

    anomaly_count = max(1, int(len(synthetic_df) * anomaly_fraction))
    anomaly_indices = rng.choice(synthetic_df.index.to_numpy(), size=anomaly_count, replace=False)

    synthetic_df.loc[anomaly_indices, "calls_offered"] *= rng.uniform(1.8, 3.0, anomaly_count)
    synthetic_df.loc[anomaly_indices, "calls_abandoned"] *= rng.uniform(2.0, 4.0, anomaly_count)
    synthetic_df.loc[anomaly_indices, "calls_answered"] *= rng.uniform(0.6, 0.9, anomaly_count)

    synthetic_df["calls_answered"] = np.minimum(synthetic_df["calls_answered"], synthetic_df["calls_offered"])
    synthetic_df["calls_abandoned"] = np.minimum(synthetic_df["calls_abandoned"], synthetic_df["calls_offered"])

    synthetic_df["unanswered_calls"] = synthetic_df["calls_offered"] - synthetic_df["calls_answered"]
    synthetic_df["answer_rate"] = np.where(
        synthetic_df["calls_offered"] > 0,
        synthetic_df["calls_answered"] / synthetic_df["calls_offered"],
        0
    )
    synthetic_df["abandon_rate"] = np.where(
        synthetic_df["calls_offered"] > 0,
        synthetic_df["calls_abandoned"] / synthetic_df["calls_offered"],
        0
    )

    if "population" in synthetic_df.columns:
        synthetic_df["calls_per_1000_population"] = np.where(
            synthetic_df["population"] > 0,
            (synthetic_df["calls_offered"] / synthetic_df["population"]) * 1000,
            0
        )

    synthetic_df.loc[anomaly_indices, "is_anomaly"] = 1

    return synthetic_df


def run_anomaly_model(df):
    feature_cols = [
        "calls_offered",
        "calls_answered",
        "calls_abandoned",
        "unanswered_calls",
        "answer_rate",
        "abandon_rate"
    ]

    if "calls_per_1000_population" in df.columns:
        feature_cols.append("calls_per_1000_population")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result_df = df.copy()
    result_df["predicted_anomaly"] = np.where(predictions == -1, 1, 0)
    result_df["anomaly_score"] = scores

    return result_df


def create_kpi_summary(df):
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Records",
            "Total Calls Offered",
            "Average Answer Rate",
            "Average Abandon Rate",
            "Predicted Anomalies"
        ],
        "Value": [
            int(len(df)),
            round(float(df["calls_offered"].sum()), 2),
            round(float(df["answer_rate"].mean()), 4),
            round(float(df["abandon_rate"].mean()), 4),
            int(df["predicted_anomaly"].sum())
        ]
    })
    return summary_df


def _empty_chart(message):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_monthly_calls(df):
    if "period" not in df.columns or df["period"].isna().all():
        return _empty_chart("Period column not available for monthly trend")

    plot_df = df.dropna(subset=["period"]).copy()
    monthly = plot_df.groupby("period", as_index=False)["calls_offered"].sum().sort_values("period")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["period"], monthly["calls_offered"], marker="o")
    ax.set_title("Monthly Trend of Total Calls Offered")
    ax.set_xlabel("Period")
    ax.set_ylabel("Total Calls Offered")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_monthly_answer_rate(df):
    if "period" not in df.columns or df["period"].isna().all():
        return _empty_chart("Period column not available for answer rate trend")

    plot_df = df.dropna(subset=["period"]).copy()
    monthly = plot_df.groupby("period", as_index=False)["answer_rate"].mean().sort_values("period")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["period"], monthly["answer_rate"], marker="o")
    ax.set_title("Monthly Average Answer Rate")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average Answer Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_monthly_abandon_rate(df):
    if "period" not in df.columns or df["period"].isna().all():
        return _empty_chart("Period column not available for abandon rate trend")

    plot_df = df.dropna(subset=["period"]).copy()
    monthly = plot_df.groupby("period", as_index=False)["abandon_rate"].mean().sort_values("period")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["period"], monthly["abandon_rate"], marker="o")
    ax.set_title("Monthly Average Abandon Rate")
    ax.set_xlabel("Period")
    ax.set_ylabel("Average Abandon Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_monthly_spikes(df):
    if "period" not in df.columns or df["period"].isna().all():
        return _empty_chart("Period column not available for anomaly spike chart")

    plot_df = df.dropna(subset=["period"]).copy()
    monthly = (
        plot_df.groupby("period", as_index=False)
        .agg({"calls_offered": "sum", "predicted_anomaly": "max"})
        .sort_values("period")
    )

    anomaly_points = monthly[monthly["predicted_anomaly"] == 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["period"], monthly["calls_offered"], marker="o", label="Calls Offered")
    ax.scatter(anomaly_points["period"], anomaly_points["calls_offered"], s=90, label="Anomaly")
    ax.set_title("Monthly Calls Offered with Anomaly Points")
    ax.set_xlabel("Period")
    ax.set_ylabel("Calls Offered")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_region_calls(df):
    if "region" not in df.columns:
        return _empty_chart("Region column not available")

    region_df = (
        df.groupby("region", as_index=False)["calls_offered"]
        .sum()
        .sort_values("calls_offered", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(region_df["region"], region_df["calls_offered"])
    ax.set_title("Top 10 Regions by Calls Offered")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Calls Offered")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_region_anomalies(df):
    if "region" not in df.columns:
        return _empty_chart("Region column not available")

    region_df = (
        df.groupby("region", as_index=False)["predicted_anomaly"]
        .sum()
        .sort_values("predicted_anomaly", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(region_df["region"], region_df["predicted_anomaly"])
    ax.set_title("Top 10 Regions by Anomaly Count")
    ax.set_xlabel("Region")
    ax.set_ylabel("Anomaly Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_calls_vs_abandon(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["calls_offered"], df["abandon_rate"], alpha=0.7)
    ax.set_title("Calls Offered vs Abandon Rate")
    ax.set_xlabel("Calls Offered")
    ax.set_ylabel("Abandon Rate")
    plt.tight_layout()
    return fig


def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(df["calls_offered"].dropna())
    ax.set_title("Box Plot of Calls Offered")
    ax.set_ylabel("Calls Offered")
    plt.tight_layout()
    return fig


def create_business_takeaway(df):
    total_anomalies = int(df["predicted_anomaly"].sum())
    avg_answer_rate = float(df["answer_rate"].mean())
    avg_abandon_rate = float(df["abandon_rate"].mean())

    return (
        f"Business takeaway\n\n"
        f"This dashboard analyses NHS service level activity and highlights unusual operational patterns.\n"
        f"The processed dataset contains {len(df)} records and the model flagged {total_anomalies} predicted anomalies.\n\n"
        f"Average answer rate is {avg_answer_rate:.2%} and average abandon rate is {avg_abandon_rate:.2%}.\n\n"
        f"Use the monthly charts to understand spikes in demand and possible drops in service performance.\n"
        f"Use the regional charts to identify where deeper operational review may be needed.\n\n"
        f"This is suitable for service monitoring and early warning analysis. "
        f"It is not patient level misuse detection because the source dataset is aggregated."
    )


def save_output_csv(df):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "processed_nhs_anomaly_output.csv")
    df.to_csv(output_path, index=False)
    return output_path


def process_nhs_file(uploaded_file, use_synthetic):
    try:
        raw_df, source_path = load_uploaded_or_default_file(uploaded_file)
        clean_df = prepare_nhs_dataset(raw_df)

        working_df = create_synthetic_dataset(clean_df) if use_synthetic else clean_df.copy()
        result_df = run_anomaly_model(working_df)

        summary_df = create_kpi_summary(result_df)
        preview_df = result_df.head(25).copy()
        anomaly_df = (
            result_df[result_df["predicted_anomaly"] == 1]
            .sort_values("anomaly_score")
            .head(50)
            .copy()
        )

        chart1 = plot_monthly_calls(result_df)
        chart2 = plot_monthly_answer_rate(result_df)
        chart3 = plot_monthly_abandon_rate(result_df)
        chart4 = plot_monthly_spikes(result_df)
        chart5 = plot_region_calls(result_df)
        chart6 = plot_region_anomalies(result_df)
        chart7 = plot_calls_vs_abandon(result_df)
        chart8 = plot_boxplot(result_df)

        business_takeaway = create_business_takeaway(result_df)
        csv_path = save_output_csv(result_df)

        status_text = (
            f"Success. Loaded source file: {os.path.basename(source_path)}. "
            f"Processed {len(result_df)} records. "
            f"Predicted anomalies: {int(result_df['predicted_anomaly'].sum())}."
        )

        return (
            status_text,
            summary_df,
            preview_df,
            anomaly_df,
            chart1,
            chart2,
            chart3,
            chart4,
            chart5,
            chart6,
            chart7,
            chart8,
            business_takeaway,
            csv_path
        )

    except Exception as e:
        error_text = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        empty_df = pd.DataFrame()

        return (
            error_text,
            empty_df,
            empty_df,
            empty_df,
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            _empty_chart("No chart available"),
            "Could not generate business takeaway because processing failed.",
            None
        )


# =========================================================
# UI
# =========================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1450px !important;
}
.section-title {
    font-weight: 600;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Anomaly Detection Dashboard") as demo:
    gr.Markdown(
        """
# Anomaly Detection Dashboard

This app uses the sample dataset 20210708-NHS-111-MDS-time-series-to-March-2021.xlsx.
Download the Open NHS dataset from:
https://www.england.nhs.uk/statistics/statistical-work-areas/iucadc-new-from-april-2021/nhs-111-minimum-data-set/nhs-111-minimum-data-set-2020-21/

It can:
- load the default NHS Excel file from your repo or folder
- accept an uploaded Excel or CSV file
- clean and prepare the dataset
- optionally create a synthetic anomaly version
- run anomaly detection
- show business friendly charts for spikes and trends
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Optional upload",
                file_types=[".xlsx", ".xls", ".csv"]
            )
            use_synthetic = gr.Checkbox(
                label="Use synthetic anomaly generation before modelling",
                value=True
            )
            run_btn = gr.Button("Run analysis", variant="primary")
            status_box = gr.Textbox(label="Status", lines=4)

        with gr.Column(scale=1):
            takeaway_box = gr.Textbox(label="Business takeaway", lines=12)
            download_output = gr.File(label="Download processed dataset")

    with gr.Tabs():
        with gr.Tab("Summary"):
            summary_table = gr.Dataframe(label="KPI Summary", interactive=False)
            preview_table = gr.Dataframe(label="Processed Data Preview", interactive=False)
            anomaly_table = gr.Dataframe(label="Top Predicted Anomalies", interactive=False)

        with gr.Tab("Trends"):
            monthly_calls_plot = gr.Plot(label="Monthly Calls Trend")
            answer_rate_plot = gr.Plot(label="Monthly Answer Rate")
            abandon_rate_plot = gr.Plot(label="Monthly Abandon Rate")
            anomaly_spike_plot = gr.Plot(label="Monthly Calls with Anomaly Points")

        with gr.Tab("Regions"):
            region_calls_plot = gr.Plot(label="Top Regions by Calls Offered")
            region_anomaly_plot = gr.Plot(label="Top Regions by Anomaly Count")

        with gr.Tab("Operational Insights"):
            calls_vs_abandon_plot = gr.Plot(label="Calls Offered vs Abandon Rate")
            calls_boxplot = gr.Plot(label="Box Plot of Calls Offered")

    run_btn.click(
        fn=process_nhs_file,
        inputs=[file_input, use_synthetic],
        outputs=[
            status_box,
            summary_table,
            preview_table,
            anomaly_table,
            monthly_calls_plot,
            answer_rate_plot,
            abandon_rate_plot,
            anomaly_spike_plot,
            region_calls_plot,
            region_anomaly_plot,
            calls_vs_abandon_plot,
            calls_boxplot,
            takeaway_box,
            download_output
        ]
    )

demo.launch()
