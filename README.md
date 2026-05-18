# 🛡 SentinelIQ — Predictive Maintenance Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

SentinelIQ is a machine learning-powered predictive maintenance dashboard built with Python and Streamlit. It ingests raw machine sensor data, runs it through trained ensemble models, and surfaces actionable failure-risk intelligence — helping maintenance teams act before a breakdown, not after.

The platform classifies each machine into one of four risk tiers (Low → Critical), ranks predicted failures by urgency, and provides model comparison tools, feature-importance analysis, and exportable maintenance schedules — all in a single dark-mode Streamlit interface.

---

## Features

- **Four-tier risk classification** — Each machine is scored with a failure probability and bucketed into Low, Medium, High, or Critical risk, with recommended actions and schedule-by dates attached automatically.
- **Dual-model comparison** — Switch between Random Forest and Gradient Boosting in the sidebar; compare their Recall, F1-Score, ROC-AUC, Precision, and cross-validated F1 side-by-side in bar charts and a radar plot.
- **Real-time sensor data analysis** — Upload any CSV of machine readings and receive instant predictions with confidence scores, row-level colour coding, and a filterable prediction table.
- **Feature impact analysis** — Visualise which sensor signals drive the model's decisions (power proxy, temperature delta, tool wear, etc.) and understand the risk direction of each feature.
- **Confidence score distribution** — A Plotly histogram shows how failure probability is spread across your fleet, letting you spot systemic stress at a glance.
- **Fleet failure rate gauge** — A live needle gauge communicates the overall fleet health status from "within safe zone" to "critical — immediate action".
- **Built-in workflow guide** — The *How It Works* tab explains predictive maintenance philosophy, key ML metrics, the processing pipeline, and risk classification logic without leaving the app.
- **Exportable results** — Download the full prediction table (including recommended actions and schedule-by dates) as a CSV for integration with maintenance systems.
- **Input validation** — User-friendly warnings for empty files, non-numeric sensor columns, and unrecognised machine types — no stack traces surfaced to end users.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard framework | [Streamlit](https://streamlit.io/) |
| ML models | [scikit-learn](https://scikit-learn.org/) — RandomForestClassifier, GradientBoostingClassifier |
| Data processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualisations | [Plotly](https://plotly.com/python/) (Express + Graph Objects) |
| Styling | Custom CSS (`assets/style.css`) loaded at runtime |
| Model persistence | Python `pickle` + `json` |
| Training dataset | [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 readings, 5 failure modes |

---

## Project Structure

```
SentinelIQ/
├── .venv/                      # Virtual environment (not committed)
├── assets/
│   └── style.css               # All dashboard CSS (loaded by app.py at runtime)
├── models/
│   ├── random_forest.pkl       # Trained Random Forest model
│   ├── gradient_boosting.pkl   # Trained Gradient Boosting model
│   ├── processed_data.pkl      # Scaler, label encoder, feature names, test split
│   └── metrics.json            # Evaluation metrics for both models
├── app.py                      # Main Streamlit application
└── requirements.txt            # Python dependencies
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- `pip` (bundled with Python)

### 1. Clone the repository

```bash
git clone https://github.com/tarun05-design/sentineliq.git
cd sentineliq
```

### 2. Create and activate a virtual environment

```bash
# Create the environment
python -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The core packages required are:

```
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
plotly
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

Once the app is running, you will see a sidebar on the left and four tabbed workspaces in the main canvas.

### Sidebar

- **Active Model** — Toggle between *Random Forest* and *Gradient Boost* using the radio selector. All predictions, charts, and live demo results update instantly.
- **Model Metrics** — Live Recall ★, F1-Score, ROC-AUC, and Precision bars for the selected model, with the best-performing model highlighted.
- **Sample Data Format** — A preview of the expected CSV schema; download a ready-to-use sample file to test the app immediately.

### Tab 1 — Predictions

1. Upload a CSV file containing machine sensor readings via the file uploader.
2. The app validates the file, runs predictions, and displays a fleet-wide alert banner (failures detected / all clear).
3. Use the **Status**, **Risk**, and **Confidence** filters to narrow the prediction table.
4. Review the styled HTML preview table (top 8 rows) or expand the full filterable dataframe.
5. Click **Export CSV** to download the complete results with recommended actions and schedule-by dates.
6. Scroll down for the confidence score histogram and fleet failure rate gauge.

> If no file is uploaded, the app shows a live demo running the selected model against the held-out test set, including Recall, failures found, and average confidence.

### Tab 2 — Model Compare

Side-by-side metric table (Accuracy, Precision, **Recall ★**, F1-Score, ROC-AUC, CV F1) with per-row winner badges, a grouped bar chart, a radar chart, and (if data is uploaded) confusion matrices and ROC curves for both models.

### Tab 3 — Feature Impact

Bar chart of feature importances from the selected model, a detailed feature insight grid explaining the risk direction of each signal, and a reference table of all engineered and raw features used during training.

### Tab 4 — How It Works

An in-app explainer covering predictive vs reactive vs scheduled maintenance, why ML outperforms rule-based approaches, key metrics (with Recall prioritised for maintenance contexts), and a five-step intelligence workflow diagram.

---

## How It Works

### Input Format

Upload a CSV with the following columns (column names are normalised — bracket suffixes and capitalisation are handled automatically):

| Column | Type | Description |
|---|---|---|
| `Type` | String | Machine quality grade: `L` (Low), `M` (Medium), or `H` (High) |
| `Air temperature` | Float | Ambient temperature reading (Kelvin) |
| `Process temperature` | Float | Process temperature reading (Kelvin) |
| `Rotational speed` | Integer | Spindle speed (RPM) |
| `Torque` | Float | Applied torque (Nm) |
| `Tool wear` | Integer | Cumulative tool wear (minutes) |

### Feature Engineering

Before prediction, two derived features are computed from the raw inputs:

- **Temperature Delta (Δ)** — `Process temperature − Air temperature`. A large gap signals poor heat dissipation and is a leading indicator of thermal failure.
- **Power Proxy** — `Torque × Rotational speed`. Approximates real mechanical power; sudden spikes correlate with stress events before sensor alarms trigger.

### Risk Classification

Each machine receives a **failure probability** (0–100%) from the model's `predict_proba` output. This is then mapped to a risk tier:

| Probability | Risk Tier | Recommended Action |
|---|---|---|
| 0 – 25% | 🟢 **Low** | Routine check |
| 26 – 60% | 🟡 **Medium** | Plan maintenance this week |
| 61 – 85% | 🟠 **High** | Schedule within 48 hours |
| > 85% | 🔴 **Critical** | Stop and inspect immediately |

---

## Model Details

### Random Forest

An ensemble of decision trees trained with bootstrap aggregation. Naturally robust to feature scale differences and provides reliable feature importance rankings. Well suited to the tabular, mixed-type sensor data in this dataset.

### Gradient Boosting

A sequential boosting ensemble that corrects the errors of prior trees. Typically achieves marginally higher Recall on imbalanced datasets when tuned well, at the cost of longer training time.

### Metric Philosophy

Both models are evaluated on **Recall as the primary metric**. In a maintenance context, a false negative (missed failure) causes unplanned downtime and emergency repair costs — far more expensive than a false positive (unnecessary inspection). Precision, F1-Score, and ROC-AUC are tracked alongside Recall for a complete picture.

Cross-validated F1 (5-fold, reported as mean ± std) guards against overfitting on the held-out test split.

All evaluation results are persisted to `models/metrics.json` in this structure:

```json
{
  "random_forest": {
    "accuracy": 0.0000,
    "precision": 0.0000,
    "recall": 0.0000,
    "f1": 0.0000,
    "roc_auc": 0.0000,
    "cv_f1_mean": 0.0000,
    "cv_f1_std": 0.0000
  },
  "gradient_boosting": { ... },
  "best_model": "Random Forest"
}
```

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`).
2. Make your changes, ensuring existing behaviour is preserved (verify the dashboard starts cleanly with `streamlit run app.py`).
3. Add or update docstrings and inline comments where relevant.
4. Open a pull request with a clear description of the change and its motivation.

**Areas where contributions are especially valued:**

- Additional model types (XGBoost, LightGBM, neural networks)
- SHAP-based explainability panel for individual prediction explanations
- Support for multi-label failure mode classification (the AI4I dataset contains five failure sub-types)
- Automated hyperparameter tuning pipeline
- Unit tests for `predict_failures` and the feature engineering logic

---

## License

This project is released under the [MIT License](LICENSE). The AI4I 2020 dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and is used here for research and demonstration purposes.
