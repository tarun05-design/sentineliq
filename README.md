# 🛡️ SentinelIQ — Predictive Maintenance Intelligence Platform

Real-time industrial sensor analytics and failure-risk intelligence powered by ensemble machine learning.

SentinelIQ is a machine learning dashboard built with Python and Streamlit. It ingests raw industrial sensor readings, processes physical stress signals ($\Delta T$ and Mechanical Power), and evaluates real-time operational risk through ensemble models to prevent costly equipment breakdowns before they occur.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-sentineliq.streamlit.app-46E3B7?style=for-the-badge&logo=streamlit&logoColor=black)](https://sentineliq.streamlit.app)

![python](https://img.shields.io/badge/python-3.9+-blue?style=flat) ![streamlit](https://img.shields.io/badge/streamlit-1.32+-red?style=flat) ![scikit-learn](https://img.shields.io/badge/model-random%20forest%20%7C%20gradient%20boosting-orange?style=flat) ![plotly](https://img.shields.io/badge/charts-plotly-purple?style=flat) ![license](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## ⚡ Highlights & Business Impact

- **Recall-First Optimization**: Engineered specifically for high-stakes maintenance environments where false negatives (unplanned downtime) carry massive costs compared to routine inspections.
- **Dynamic 4-Tier Risk Scoring**: Classifies machinery into **Low (0-25%)**, **Medium (26-60%)**, **High (61-85%)**, and **Critical (>85%)** operational risk tiers with automated inspection scheduling.
- **Dual Ensemble Model Benchmarking**: Built-in side-by-side comparison between **Random Forest** and **Gradient Boosting** models with interactive metrics (Recall, Precision, F1-Score, ROC-AUC, CV-F1).
- **Domain Feature Engineering**: Transforms raw sensor readings into physical stress indicators like **Temperature Delta ($\Delta T$)** and **Mechanical Power Proxy ($P \propto \tau \times \omega$)**.

---

## 🛠️ Tech Stack & Architecture

| Category | Technology / Library | Usage & Purpose |
|---|---|---|
| **App Framework** | [Streamlit](https://streamlit.io/) | Interactive dark-mode dashboard engine & live state management |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/), [imbalanced-learn](https://imbalanced-learn.org/) | Random Forest & Gradient Boosting classifiers with class-imbalance handling |
| **Data Processing** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | Real-time sensor stream ingestion, transformation, & feature extraction |
| **Data Visualization** | [Plotly](https://plotly.com/python/) | Interactive radar plots, risk distribution histograms, & gauge charts |
| **Training Dataset** | [UCI AI4I 2020 Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 10,000 real-world industrial machine operational readings |

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A[Raw Machine Sensors\nAir Temp, Process Temp, Speed, Torque, Wear] --> B[Input Validation & Normalization]
    B --> C[Feature Engineering Engine]
    C --> D1["Temp Delta (ΔT = Process - Air)"]
    C --> D2["Power Proxy (Torque × RPM)"]
    D1 & D2 --> E[Scaler & Preprocessing Pipeline]
    E --> F{Model Selector}
    F -->|Ensemble 1| G1[Random Forest Classifier]
    F -->|Ensemble 2| G2[Gradient Boosting Classifier]
    G1 & G2 --> H[Probability Score Generation]
    H --> I[Risk Tier Bucketizer\nLow | Medium | High | Critical]
    I --> J[Streamlit Dashboard & Maintenance Schedule Export]
```

---

## ⚡ Core Features

### 1. Fleet Real-Time Diagnostics
Upload any industrial telemetry CSV or launch the interactive live demo dataset. SentinelIQ automatically cleans, transforms, and runs predictions, generating row-level risk highlighting, failure alerts, and a summary breakdown.

### 2. Dual-Model Interactive Comparison
Compare models across 6 statistical metrics. Inspect real-time radar performance charts, decision boundaries, and cross-validated F1 variance ($5$-fold CV).

### 3. Feature Impact & Signal Analysis
Visualize feature importances and directional risk drivers. Understand how thermal stress and mechanical torque influence machine health.

### 4. Automated Maintenance Export
Download clean CSV reports complete with risk levels, confidence scores, recommended engineering interventions, and suggested maintenance deadlines.

---

## 📁 Repository Structure

```tree
SentinelIQ/
├── assets/
│   └── style.css               # Runtime custom dashboard CSS styling
├── models/
│   ├── random_forest.pkl       # Trained Random Forest classifier binary
│   ├── gradient_boosting.pkl   # Trained Gradient Boosting classifier binary
│   ├── processed_data.pkl      # Saved scaler, encoders, & test splits
│   └── metrics.json            # Model evaluation & benchmark metadata
├── app.py                      # Core Streamlit application entry point
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.9+ installed
- `pip` package manager

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/tarun05-design/sentineliq.git
cd sentineliq

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Application
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 👤 Author & Connect

**Tarun P** — Machine Learning & Full Stack Developer
- 🌐 Portfolio: [tarun-portfolio.vercel.app](https://tarun-portfolio.vercel.app)
- 🐙 GitHub: [@tarun05-design](https://github.com/tarun05-design)
- 📧 Email: [tarunparthasarathy65@gmail.com](mailto:tarunparthasarathy65@gmail.com)
