# Anomaly Detection Dashboard

This project is a Gradio-based dashboard for detecting unusual patterns in NHS operational data. It loads time-series data from Excel or CSV, prepares the dataset, runs anomaly detection with `IsolationForest`, and presents business-friendly summaries, charts, and downloadable output.

GitHub repository:
`https://github.com/aitiwari/anomaly_detection.git`

## Features

- Loads the default NHS Excel file from the project folder or accepts an uploaded file
- Supports `.xlsx`, `.xls`, and `.csv` inputs
- Cleans and standardizes column names automatically
- Builds operational metrics such as answer rate, abandon rate, and unanswered calls
- Optionally injects synthetic anomalies for demo or testing scenarios
- Detects anomalies using scikit-learn's `IsolationForest`
- Shows KPI tables, monthly trend charts, regional views, and operational insight plots
- Exports the processed dataset as CSV

## Project Structure

- `app.py` - main Gradio application
- `requirements.txt` - Python dependencies
- `20210708-NHS-111-MDS-time-series-to-March-2021.xlsx` - default sample dataset used by the app if present

## Requirements

- Python 3.10+ recommended
- A virtual environment is recommended

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run Locally

Clone the repository:

```powershell
git clone https://github.com/aitiwari/anomaly_detection.git
cd anomaly_detection
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the app:

```powershell
python app.py
```

After launch, Gradio will print a local URL in the terminal that you can open in your browser.

## Data Input

The app uses the sample dataset `20210708-NHS-111-MDS-time-series-to-March-2021.xlsx` and will automatically look for:

- `20210708-NHS-111-MDS-time-series-to-March-2021.xlsx`
- `./20210708-NHS-111-MDS-time-series-to-March-2021.xlsx`
- `./data/20210708-NHS-111-MDS-time-series-to-March-2021.xlsx`

Download the Open NHS dataset from:
`https://www.england.nhs.uk/statistics/statistical-work-areas/iucadc-new-from-april-2021/nhs-111-minimum-data-set/nhs-111-minimum-data-set-2020-21/`

If no default file is found, you can upload a supported Excel or CSV file through the UI.

The application expects operational fields related to:

- calls offered
- calls answered
- calls abandoned
- period or date
- region or provider level identifiers when available

## How It Works

1. The input file is loaded and standardized.
2. Key service metrics are derived from the source data.
3. Optional synthetic anomalies can be added for experimentation.
4. The dataset is scaled and scored with `IsolationForest`.
5. Results are displayed in tables, charts, and a plain-language business summary.

## Output

The dashboard provides:

- status summary
- KPI table
- processed data preview
- top predicted anomaly records
- monthly trend and anomaly charts
- regional charts
- downloadable processed CSV output

## Notes

- This project analyzes aggregated operational data, not patient-level data.
- The anomaly output is best used as an early-warning or service-monitoring aid.
- Synthetic anomaly generation is useful for demos, testing, and validating the dashboard flow.

## License

This project includes an MIT license. See `LICENSE` for details.
