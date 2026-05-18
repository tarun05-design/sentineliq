"""
=============================================================
SENTINELIQ — Predictive Maintenance Intelligence Platform
=============================================================
HOW TO RUN:
  1. Make sure models/ folder has the .pkl and .json files
     (run 01_preprocess_eda.py then 02_train_models.py first)
  2. pip install streamlit pandas numpy scikit-learn plotly
  3. streamlit run app.py
  4. Open http://localhost:8501 in your browser
=============================================================
Layout: Sidebar + Main Canvas  ·  v3 — matches UI mockup exactly
  Alert → Inline metrics → Tabs → Filters → Table → Export
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelIQ — Predictive Maintenance",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# DESIGN SYSTEM — CSS
# Dark charcoal/graphite palette matching the mockup:
#   Background  #0d0f12   (near-black, slightly warm)
#   Sidebar     #111318   (cool dark)
#   Panels      #181b22   (card surfaces)
#   Borders     #22273a   (subtle separators)
#   Accent blue #2563eb / #3b82f6
#   Success     #16a34a / #22c55e
#   Danger      #dc2626 / #ef4444
# ─────────────────────────────────────────────────────────
st.markdown(f"<style>{open('assets/style.css', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

# ── Mobile viewport meta (injected once) ──────────────────
# Ensures the browser respects our @media breakpoints instead
# of zooming a desktop-width viewport into a tiny phone screen.
st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5">',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models, errors = {}, []
    for key, path in [
        ("random_forest",     "models/random_forest.pkl"),
        ("gradient_boosting", "models/gradient_boosting.pkl"),
    ]:
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            errors.append(f"{path} not found")

    try:
        with open("models/processed_data.pkl", "rb") as f:
            p = pickle.load(f)
            models.update({
                "scaler":        p["scaler"],
                "label_encoder": p["label_encoder"],
                "feature_names": p["feature_names"],
                "X_test":        p["X_test"],
                "y_test":        p["y_test"],
            })
    except FileNotFoundError:
        errors.append("models/processed_data.pkl not found")

    try:
        with open("models/metrics.json") as f:
            models["metrics"] = json.load(f)
    except FileNotFoundError:
        errors.append("models/metrics.json not found")

    return models, errors


# ─────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────
def predict_failures(df_raw, models, model_key):
    df = df_raw.copy()
    df.columns = df.columns.str.replace(r"\s*\[.*?\]", "", regex=True).str.strip()
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    required = ["type", "air_temperature", "process_temperature",
                "rotational_speed", "torque", "tool_wear"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"

    if df.empty:
        return None, "The uploaded CSV is empty — please provide at least one row of sensor data."
    numeric_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    for col in numeric_cols:
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            return None, f"Non-numeric values found in column '{col}'. All sensor columns must contain numbers."
    if not df["type"].astype(str).str.upper().str.strip().isin({"L", "M", "H"}).all():
        return None, "Unknown machine type detected. The 'Type' column must contain only L, M, or H."

    df["type"]         = df["type"].str.upper().str.strip()
    df["type_encoded"] = df["type"].map({"L": 0, "M": 1, "H": 2}).fillna(1)
    df["temp_diff"]    = df["process_temperature"] - df["air_temperature"]
    df["power_proxy"]  = df["torque"] * df["rotational_speed"]

    X_scaled = models["scaler"].transform(df[models["feature_names"]])
    model    = models[model_key]
    probs    = model.predict_proba(X_scaled)[:, 1]
    preds    = model.predict(X_scaled)

    out = df_raw.copy()
    out["Failure Predicted"]   = preds
    out["Failure Probability"] = probs.round(4)
    out["Confidence (%)"]      = (probs * 100).round(1)
    out["Status"]              = pd.Series(preds).map({0: "Normal", 1: "FAILURE"}).values
    out["Risk Level"]          = pd.cut(
        probs, bins=[0, 0.25, 0.60, 0.85, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )
    return out, None


# ─────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────
_PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,15,20,0.5)",
    font=dict(color="#8899aa", size=12),
)

def plotly_layout(**kw):
    """Safely merge _PLOTLY_BASE with per-chart overrides."""
    cfg = dict(_PLOTLY_BASE)
    cfg.setdefault("margin", dict(t=45, b=30, l=20, r=20))
    cfg.update(kw)
    return cfg


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
def render_sidebar(models, metrics_available):
    with st.sidebar:
        # Brand header
        st.markdown(
            '<div class="brand-header">'
            '<div class="brand-icon">&#x26E8;</div>'
            '<div>'
            '<div class="brand-title">SentinelIQ</div>'
            '<div class="brand-subtitle">Fleet Intelligence</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Active Model ────────────────────────────────
        st.markdown('<span class="sb-label">Active Model</span>', unsafe_allow_html=True)

        # Single clean model selector
        model_choice = st.radio(
            "Select Model",
            ["Random Forest", "Gradient Boost"],
        )
        model_key = "random_forest" if "Forest" in model_choice else "gradient_boosting"

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

        # ── Model Metrics ───────────────────────────────
        if metrics_available:
            st.markdown('<span class="sb-label">Model Metrics</span>', unsafe_allow_html=True)
            m = models["metrics"][model_key]
            defs = [
                ("Recall",    m["recall"],    True),
                ("F1-Score",  m["f1"],        False),
                ("ROC-AUC",   m["roc_auc"],   False),
                ("Precision", m["precision"], False),
            ]
            rows = []
            for lbl, val, star in defs:
                bar_cls = "sb-bar-fill recall" if star else "sb-bar-fill"
                star_html = ' <span class="sb-metric-star">&#9733;</span>' if star else ""
                rows.append(
                    f'<div class="sb-metric">'
                    f'<div class="sb-metric-row">'
                    f'<span class="sb-metric-label">{lbl}{star_html}</span>'
                    f'<span class="sb-metric-value">{val:.3f}</span>'
                    f'</div>'
                    f'<div class="sb-bar-bg"><div class="{bar_cls}" style="width:{val*100:.1f}%"></div></div>'
                    f'</div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)

            best = models["metrics"].get("best_model", "")
            if best:
                st.markdown(
                    f'<div class="best-badge">Best overall: {best}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

        # ── Quick Data ──────────────────────────────────
        st.markdown('<span class="sb-label">Quick Data</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sb-data-card">'
            '<span>&#128196;</span>'
            '<div>'
            '<div class="sb-data-name">ai4i2020.csv &nbsp;&#183;&nbsp; 509 KB</div>'
            '<div class="sb-data-sub">Last run: just now</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

        # ── Sample CSV ──────────────────────────────────
        st.markdown('<span class="sb-label">Sample Data Format</span>', unsafe_allow_html=True)
        sample_data = {
            "Type":                ["L",    "M",    "H"   ],
            "Air temperature":     [298.1,  300.5,  299.0 ],
            "Process temperature": [308.6,  310.2,  307.8 ],
            "Rotational speed":    [1551,   1408,   1862  ],
            "Torque":              [42.8,   46.3,   25.1  ],
            "Tool wear":           [0,      143,    220   ],
        }
        st.dataframe(pd.DataFrame(sample_data), hide_index=True, use_container_width=True)
        st.download_button(
            "&#11015; Download Sample CSV",
            pd.DataFrame(sample_data).to_csv(index=False),
            "sample_sensor_data.csv", "text/csv",
            use_container_width=True,
        )

    return model_key, model_choice


# ─────────────────────────────────────────────────────────
# HELPERS — HTML COMPONENTS
# ─────────────────────────────────────────────────────────
def _kpi_strip(n_total, n_failures, failure_rate, n_critical, n_high, avg_prob):
    """Inline 4-cell metric row matching the mockup."""
    fail_cls  = "orange" if n_failures > 0 else ""
    crit_cls  = "red"    if n_critical > 0 else ""
    prob_cls  = "orange" if avg_prob > 10   else ""
    return (
        '<div class="kpi-strip">'

        '<div class="kpi-cell">'
        '<div class="kpi-lbl">Total machines</div>'
        f'<div class="kpi-num">{n_total:,}</div>'
        '<div class="kpi-sub">in uploaded dataset</div>'
        '</div>'

        '<div class="kpi-cell">'
        '<div class="kpi-lbl">Failures predicted</div>'
        f'<div class="kpi-num {fail_cls}">{n_failures:,}</div>'
        f'<div class="kpi-sub">{failure_rate:.1f}% of fleet</div>'
        '</div>'

        '<div class="kpi-cell">'
        '<div class="kpi-lbl">Critical risk</div>'
        f'<div class="kpi-num {crit_cls}">{n_critical:,}</div>'
        f'<div class="kpi-sub">{n_critical} critical &middot; {n_high} high</div>'
        '</div>'

        '<div class="kpi-cell">'
        '<div class="kpi-lbl">Avg failure prob.</div>'
        f'<div class="kpi-num {prob_cls}">{avg_prob:.1f}%</div>'
        '<div class="kpi-sub">across all machines</div>'
        '</div>'

        '</div>'
    )


def _prediction_table_html(df, max_preview=8):
    """
    Render the prediction table as HTML with row-level colour coding,
    status pills, dot indicators, confidence colouring, recommended
    action and schedule-by date for failure rows.
    Only shows max_preview rows inline; rest shown via st.dataframe below.
    """
    total = len(df)
    rows_html = []
    for _, row in df.head(max_preview).iterrows():
        status  = str(row.get("Status", ""))
        conf    = float(row.get("Conf.", 0))
        is_fail = status == "FAILURE"
        is_crit = is_fail and conf >= 85

        row_cls = "row-fail-hi" if is_crit else ("row-fail" if is_fail else "row-ok")
        risk_level = str(row.get("_risk", "Low"))
        action     = str(row.get("Action", "—"))
        sched_by   = str(row.get("Schedule By", "—"))

        if is_fail:
            pill_html = (
                f'<span class="tpill tpill-fail">'
                f'<span class="tdot tdot-r"></span>'
                f'FAILURE · {risk_level}'
                f'</span>'
            )
            conf_cls = "conf-hi"
            # Action badge colour by urgency
            act_style = {
                "Critical": "background:#3d0505;color:#f87171;border:1px solid #7f1d1d",
                "High":     "background:#1e1000;color:#fb923c;border:1px solid #92400e",
                "Medium":   "background:#1a1800;color:#fbbf24;border:1px solid #78350f",
                "Low":      "background:#0a1f10;color:#4ade80;border:1px solid #166534",
            }.get(risk_level, "background:#0a1f10;color:#4ade80;border:1px solid #166534")
            action_html = (
                f'<span style="display:inline-block;font-size:0.72rem;font-weight:700;'
                f'padding:2px 8px;border-radius:8px;{act_style};">{action}</span>'
            )
            sched_html = (
                f'<span style="font-size:0.78rem;font-weight:700;color:#60a5fa;">{sched_by}</span>'
            )
        else:
            pill_html = (
                f'<span class="tpill tpill-ok">'
                f'<span class="tdot tdot-g"></span>'
                f'Normal · {risk_level}'
                f'</span>'
            )
            conf_cls   = "conf-ok"
            action_html = '<span style="font-size:0.75rem;color:#3a4a5a;">—</span>'
            sched_html  = '<span style="font-size:0.75rem;color:#3a4a5a;">—</span>'

        conf_str = f"{conf:.0f}%"
        rows_html.append(
            f'<tr class="{row_cls}">'
            f'<td>{int(row.get("Row", 0))}</td>'
            f'<td>{row.get("Type", "")}</td>'
            f'<td>{int(row.get("RPM", 0))}</td>'
            f'<td>{float(row.get("Torque", 0)):.1f}</td>'
            f'<td>{int(row.get("Wear", 0))}</td>'
            f'<td>{pill_html}</td>'
            f'<td class="{conf_cls}">{conf_str}</td>'
            f'<td>{action_html}</td>'
            f'<td>{sched_html}</td>'
            f'</tr>'
        )

    more = total - max_preview
    more_row = (
        f'<tr><td colspan="9" class="table-more">+ {more:,} more rows — download CSV to view all</td></tr>'
        if more > 0 else ""
    )

    return (
        '<table class="pred-table">'
        '<thead><tr>'
        '<th>Row</th><th>Type</th><th>RPM</th>'
        '<th>Torque</th><th>Wear</th>'
        '<th>Status</th><th>Conf.</th>'
        '<th>Recommended Action</th><th>Schedule By</th>'
        '</tr></thead>'
        '<tbody>'
        + "".join(rows_html)
        + more_row
        + '</tbody></table>'
    )


def _status_indicator_panel(failure_rate: float):
    """Three-column legend panel: Machine Status | Confidence Bands | Mini Gauge."""
    pct = min(failure_rate, 100)
    zone = (
        "Within safe zone"    if failure_rate < 10 else
        "Elevated — review"   if failure_rate < 30 else
        "Critical — act now"
    )
    # Build as single concatenated string — no HTML comments, no f-string in triple quotes
    html = (
        '<div class="si-panel">'

        '<div>'
        '<div class="si-title">Machine Status</div>'
        '<div class="si-row"><span class="dot dot-g"></span><span class="sp sp-n">Normal</span><span class="sp-txt">Low risk</span></div>'
        '<div class="si-row"><span class="dot dot-a"></span><span class="sp sp-m">Medium</span><span class="sp-txt">Monitor</span></div>'
        '<div class="si-row"><span class="dot dot-r"></span><span class="sp sp-h">High</span><span class="sp-txt">Urgent</span></div>'
        '<div class="si-row"><span class="dot dot-d"></span><span class="sp sp-c">Critical</span><span class="sp-txt">Immediate</span></div>'
        '</div>'

        '<div>'
        '<div class="si-title">Confidence Bands</div>'
        '<div class="band-row"><span class="band-sw bw-g"></span><span>0&ndash;25% &mdash; Reliable Normal</span></div>'
        '<div class="band-row"><span class="band-sw bw-a"></span><span>26&ndash;60% &mdash; Review Needed</span></div>'
        '<div class="band-row"><span class="band-sw bw-r"></span><span>61&ndash;85% &mdash; High Confidence</span></div>'
        '<div class="band-row"><span class="band-sw bw-d"></span><span>85&ndash;100% &mdash; Act Immediately</span></div>'
        '</div>'

        '<div>'
        '<div class="si-title">Gauge Redesign</div>'
        '<div class="mg-labels"><span>0%</span><span>50%</span><span>100%</span></div>'
        f'<div class="mg-bar"><div class="mg-needle" style="left:{pct:.1f}%"></div></div>'
        f'<div class="mg-num">{failure_rate:.1f}%</div>'
        f'<div class="mg-sub">Fleet failure rate &mdash; {zone}</div>'
        '</div>'

        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────
def main():
    models, load_errors = load_models()

    if load_errors:
        st.error("Models not found. Run the training scripts first.")
        with st.expander("Setup Instructions", expanded=True):
            st.code("""
pip install pandas numpy scikit-learn imbalanced-learn plotly streamlit
python 01_preprocess_eda.py
python 02_train_models.py
streamlit run app.py
            """, language="bash")
        st.stop()

    models_loaded     = "random_forest" in models and "gradient_boosting" in models
    metrics_available = "metrics" in models

    model_key, model_choice = render_sidebar(models, metrics_available)

    st.markdown(
        '<h1 class="page-title">Predictive Maintenance Intelligence</h1>'
        '<p style="font-size:0.95rem;color:#a3a3a3;margin:0 0 1.25rem;">Fleet monitoring, risk analytics, and machine failure prediction dashboard</p>',
        unsafe_allow_html=True,
    )

    # ── Four workspace tabs ──────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "&#9678;  Predictions",
        "&#9638;  Model Compare",
        "&#10150;  Feature Impact",
        "&#9633;  How It Works",
    ])


    # ════════════════════════════════════════════════════
    # TAB 1 — PREDICTIONS
    # ════════════════════════════════════════════════════
    with tab1:

        # ── Compact upload + required-columns row ────────
        # On desktop: [3, 1] split. On mobile Streamlit stacks
        # columns naturally when the viewport is narrow.
        up_col, req_col = st.columns([3, 1])
        with up_col:
            uploaded_file = st.file_uploader(
                "Upload CSV", type=["csv"],
                label_visibility="collapsed",
            )
        with req_col:
            st.markdown(
                '<div class="req-box">'
                '<strong>&#128204; Required columns</strong>'
                'Type (L/M/H) &middot; Air temperature &middot; Process temperature'
                ' &middot; Rotational speed &middot; Torque &middot; Tool wear'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── No file uploaded ─────────────────────────────
        if not uploaded_file:
            st.markdown("""
<div class="panel-card" style="margin-top:0.5rem;">
  <div class="panel-card-title">
    <div class="panel-card-icon">&#9432;</div>
    Ready to Analyse
  </div>
  <p>Upload a CSV of machine sensor readings above to get instant failure predictions. Download the <strong>Sample CSV</strong> from the sidebar to test immediately.</p>
  <div class="info-pills">
    <span class="info-pill">Type (L/M/H)</span>
    <span class="info-pill">Air Temperature</span>
    <span class="info-pill">Process Temperature</span>
    <span class="info-pill">Rotational Speed</span>
    <span class="info-pill">Torque</span>
    <span class="info-pill">Tool Wear</span>
  </div>
</div>
""", unsafe_allow_html=True)

            if models_loaded:
                X_test = models["X_test"]
                y_test = models["y_test"]
                y_prob = models[model_key].predict_proba(X_test)[:, 1]
                y_pred = models[model_key].predict(X_test)
                _demo_n_fail = int(sum(y_pred))
                _true_pos = sum(1 for a, b in zip(y_test, y_pred) if a == 1 and b == 1)
                _actual_pos = sum(1 for a in y_test if a == 1)
                _demo_recall = (_true_pos / _actual_pos * 100) if _actual_pos > 0 else 0.0
                _demo_avg_conf = float(y_prob.mean() * 100)
                st.markdown(f"""
<div class="panel-card accent-green" style="margin-top:0.4rem;">
  <div class="panel-card-title">
    <div class="panel-card-icon gi-green">&#128202;</div>
    Live Demo — Held-Out Test Set
  </div>
  <p>Running <strong>{model_choice}</strong> on the held-out validation data. Upload your own CSV above to analyse real fleet data.</p>
  <div class="stat-tiles stat-tiles-3col">
    <div class="stat-tile">
      <div class="stat-tile-icon ti-green">&#9650;</div>
      <div>
        <div class="stat-tile-label">Recall ★</div>
        <div class="stat-tile-value">{_demo_recall:.1f}%</div>
        <div class="stat-tile-sub">failures caught</div>
      </div>
    </div>
    <div class="stat-tile">
      <div class="stat-tile-icon ti-red">&#9888;</div>
      <div>
        <div class="stat-tile-label">Failures found</div>
        <div class="stat-tile-value">{_demo_n_fail:,}</div>
        <div class="stat-tile-sub">of {len(y_pred):,} samples</div>
      </div>
    </div>
    <div class="stat-tile">
      <div class="stat-tile-icon">&#127919;</div>
      <div>
        <div class="stat-tile-label">Avg Confidence</div>
        <div class="stat-tile-value">{_demo_avg_conf:.1f}%</div>
        <div class="stat-tile-sub">across all samples</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
                demo_df = pd.DataFrame({
                    "Sample":         range(1, len(y_test) + 1),
                    "Actual":         ["Failure" if y == 1 else "Normal" for y in y_test],
                    "Predicted":      ["Failure" if y == 1 else "Normal" for y in y_pred],
                    "Confidence (%)": (y_prob * 100).round(1),
                    "Correct":        ["\u2713" if a == b else "\u2717" for a, b in zip(y_test, y_pred)],
                })
                st.dataframe(demo_df.head(60), use_container_width=True, height=360, hide_index=True)

            _status_indicator_panel(0.0)
            return

        # ── Read CSV ─────────────────────────────────────
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # ── Run predictions ──────────────────────────────
        with st.spinner("Analysing sensor data…"):
            results, error = predict_failures(df_raw, models, model_key)
        if error:
            st.error(error)
            return

        # ── Derived stats ────────────────────────────────
        n_total      = len(results)
        n_failures   = int(results["Failure Predicted"].sum())
        failure_rate = n_failures / n_total * 100
        avg_prob     = results["Failure Probability"].mean() * 100
        n_critical   = int((results["Risk Level"] == "Critical").sum())
        n_high       = int((results["Risk Level"] == "High").sum())
        n_high_crit  = n_critical + n_high

        # ── Alert banner (full width, prominent) ─────────
        if n_failures > 0:
            st.markdown(
                '<div class="alert-critical">'
                '<span class="alert-icon">&#9888;</span>'
                f'<span><strong>{n_failures:,} failure{"s" if n_failures!=1 else ""} detected</strong>'
                f' across {n_total:,} machines &mdash; {failure_rate:.1f}% of fleet requires attention</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-ok">'
                '<span class="alert-icon">&#10003;</span>'
                f'<span>All <strong>{n_total:,} machines</strong> operating normally &mdash; '
                f'no failures predicted by {model_choice}.</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Inline KPI strip ─────────────────────────────
        st.markdown(
            _kpi_strip(n_total, n_failures, failure_rate, n_critical, n_high, avg_prob),
            unsafe_allow_html=True,
        )

        # ── Prediction table + filters ────────────────────
        import datetime as _dt
        today = _dt.date.today()

        # Scheduling rules based on urgency
        _deadline_days = {"Critical": 0, "High": 1, "Medium": 7, "Low": 30}
        _action_map = {
            "Critical": "🔴 Stop & inspect now",
            "High":     "🟠 Schedule within 48h",
            "Medium":   "🟡 Plan this week",
            "Low":      "🟢 Routine check",
        }

        def _sched_date(risk):
            days = _deadline_days.get(risk, 30)
            return (today + _dt.timedelta(days=days)).strftime("%d %b %Y")

        def _col(df, variants):
            for c in variants:
                if c in df.columns:
                    return df[c]
            return pd.Series(["-"] * len(df))

        risk_vals = results["Risk Level"].astype(str).values
        clean_df = pd.DataFrame({
            "Row":       range(1, n_total + 1),
            "Type":      _col(df_raw, ["Type", "type"]),
            "RPM":       pd.to_numeric(_col(df_raw, ["Rotational speed [rpm]", "rotational_speed", "Rotational speed"]), errors="coerce").fillna(0).astype(int),
            "Torque":    pd.to_numeric(_col(df_raw, ["Torque [Nm]", "torque", "Torque"]), errors="coerce").fillna(0).round(1),
            "Wear":      pd.to_numeric(_col(df_raw, ["Tool wear [min]", "tool_wear", "Tool wear"]), errors="coerce").fillna(0).astype(int),
            "Status":    results["Status"].values,
            "Conf.":     results["Confidence (%)"].values,
            "_risk":     risk_vals,
            "Action":    [_action_map.get(r, "🟢 Routine check") for r in risk_vals],
            "Schedule By": [_sched_date(r) if results["Failure Predicted"].values[i] == 1 else "—"
                            for i, r in enumerate(risk_vals)],
        })

        # Filter controls
        # 4-column layout on desktop; Streamlit collapses gracefully on mobile
        # via the flex-wrap rule in our mobile CSS.
        fc1, fc2, fc3, fc4 = st.columns([1.1, 1, 1.4, 0.6])
        with fc1:
            f_status = st.selectbox("Status", ["All", "Failures Only", "Normal Only"], label_visibility="collapsed")
        with fc2:
            f_risk   = st.selectbox("Risk", ["All Risk", "Critical", "High", "Medium", "Low"], label_visibility="collapsed")
        with fc3:
            f_conf   = st.slider("Confidence Level (%)", 0, 100, (0, 100), step=5)
        with fc4:
            st.markdown("<br>", unsafe_allow_html=True)

        # Apply filters
        filt = clean_df.copy()
        if f_status == "Failures Only":
            filt = filt[filt["Status"] == "FAILURE"]
        elif f_status == "Normal Only":
            filt = filt[filt["Status"] == "Normal"]
        if f_risk != "All Risk":
            filt = filt[filt["_risk"] == f_risk]
        filt = filt[(filt["Conf."] >= f_conf[0]) & (filt["Conf."] <= f_conf[1])]

        # Filter row summary (pill tags + count)
        active_tags = []
        if f_status == "Failures Only":
            active_tags.append('<span class="ftag ftag-fail">&#9888; Failures only</span>')
        if f_risk != "All Risk":
            active_tags.append(f'<span class="ftag ftag-neut">&#9661; {f_risk} risk</span>')
        active_tags.append(f'<span class="ftag ftag-neut">Conf: {f_conf[0]}&ndash;{f_conf[1]}%</span>')
        tags_html = (
            '<div class="filter-row">'
            '<span class="filter-lbl">Filter:</span>'
            + "".join(active_tags)
            + f'<span class="shown-count">{len(filt):,} shown</span>'
            + '</div>'
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        # HTML preview table (top 8 rows, styled)
        # Wrapped in .pred-table-wrap for horizontal scroll on mobile
        st.markdown(
            '<div class="pred-table-wrap">'
            + _prediction_table_html(filt, max_preview=8)
            + '</div>',
            unsafe_allow_html=True,
        )

        # Full filterable dataframe for power users (collapsed by default)
        with st.expander(f"&#128065; Full table ({len(filt):,} rows)", expanded=False):
            def _highlight(row):
                if row["Status"] == "FAILURE":
                    return ["background-color:#2a0808; color:#fca5a5"] * len(row)
                return ["background-color:#0a140d; color:#86efac"] * len(row)

            display_cols = [c for c in filt.columns if c != "_risk"]
            st.dataframe(
                filt[display_cols].style.apply(_highlight, axis=1)
                    .format({"Conf.": "{:.1f}%", "Torque": "{:.1f}"}),
                use_container_width=True, height=420, hide_index=True,
            )

        # ── Loaded bar + Export CSV ──────────────────────
        lb1, lb2 = st.columns([3, 1])
        with lb1:
            st.markdown(
                f'<div class="loaded-pill">'
                f'&#9989; Loaded {n_total:,} &times; {len(df_raw.columns)} columns'
                f'</div>',
                unsafe_allow_html=True,
            )
        with lb2:
            # Export includes Action + Schedule By columns
            export_df = results.copy()
            export_df["Recommended Action"] = clean_df["Action"].values
            export_df["Schedule By"]        = clean_df["Schedule By"].values
            st.download_button(
                "&#11015; Export CSV",
                export_df.to_csv(index=False),
                f"sentineliq_{model_choice.lower().replace(' ', '_')}.csv",
                "text/csv",
                use_container_width=True,
                type="primary",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence histogram ─────────────────────────
        st.markdown("""
<div class="panel-card" style="margin-bottom:0.5rem;">
  <div class="panel-card-title">
    <div class="panel-card-icon">&#128202;</div>
    Confidence Score Distribution
  </div>
  <p>How the model's predicted failure probability is spread across your fleet. A healthy fleet clusters near 0%; a stressed fleet shifts rightward toward red.</p>
</div>
""", unsafe_allow_html=True)
        fig_hist = px.histogram(
            results, x="Confidence (%)", color="Status", nbins=40,
            color_discrete_map={"Normal": "#16a34a", "FAILURE": "#dc2626"},
        )
        fig_hist.update_layout(**plotly_layout(
            height=240, legend_title="",
            margin=dict(t=20, b=30, l=10, r=10),
            xaxis_title="Confidence (%)", yaxis_title="Machine count",
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Fleet Failure Rate gauge ─────────────────────
        needle = min(failure_rate, 100)
        zone_lbl = (
            "Within safe zone"       if failure_rate < 10 else
            "Elevated — review fleet" if failure_rate < 30 else
            "Critical — immediate action"
        )
        st.markdown(
            f'<div class="panel-card accent-red" style="margin-bottom:0.5rem;">'
            f'<div class="panel-card-title">'
            f'<div class="panel-card-icon gi-red">&#128207;</div>'
            f'Fleet Failure Rate'
            f'</div>'
            f'<div class="gauge-wrap">'
            f'<div class="gauge-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'
            f'<div class="gauge-bar"><div class="gauge-needle" style="left:{needle:.1f}%"></div></div>'
            f'<div class="gauge-readout"><span class="gauge-num">{failure_rate:.1f}%</span>'
            f'<span class="gauge-desc">Fleet failure rate &mdash; {zone_lbl}</span></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Status Indicator System panel ────────────────
        _status_indicator_panel(failure_rate)

    # ════════════════════════════════════════════════════
    # TAB 2 — MODEL COMPARE
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="sec-hdr">Model Performance Comparison</div>', unsafe_allow_html=True)

        if not metrics_available:
            st.markdown("""
<div class="panel-card accent-amber">
  <div class="panel-card-title">
    <div class="panel-card-icon gi-amber">&#9888;</div>
    Metrics Unavailable
  </div>
  <p>Model metrics not found. Run the training scripts first to generate metrics.json.</p>
  <div class="info-pills">
    <span class="info-pill amber">python 01_preprocess_eda.py</span>
    <span class="info-pill amber">python 02_train_models.py</span>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            rf_m = models["metrics"]["random_forest"]
            gb_m = models["metrics"]["gradient_boosting"]
            best = models["metrics"].get("best_model", "")

            # ── Styled comparison table ──────────────────
            metric_rows = [
                ("Accuracy",          rf_m['accuracy'],  gb_m['accuracy'],  False),
                ("Precision",         rf_m['precision'], gb_m['precision'], False),
                ("Recall",            rf_m['recall'],    gb_m['recall'],    True),
                ("F1-Score",          rf_m['f1'],        gb_m['f1'],        False),
                ("ROC-AUC",           rf_m['roc_auc'],   gb_m['roc_auc'],   False),
            ]
            rows_html = []
            for lbl, rv, gv, star in metric_rows:
                rf_best = rv >= gv
                r_cls = "best" if rf_best else ""
                g_cls = "best" if not rf_best else ""
                star_cls = "star-row" if star else ""
                rf_badge = '<span class="winner-badge">&#9654; Winner</span>' if rf_best else ""
                gb_badge = '<span class="winner-badge">&#9654; Winner</span>' if not rf_best else ""
                rows_html.append(
                    f'<tr>'
                    f'<td class="metric-name {star_cls}">{lbl}</td>'
                    f'<td class="{r_cls}">{rv:.4f}{rf_badge}</td>'
                    f'<td class="{g_cls}">{gv:.4f}{gb_badge}</td>'
                    f'</tr>'
                )
            cv_row = (
                f'<tr><td class="metric-name">CV F1 (mean ± std)</td>'
                f'<td>{rf_m["cv_f1_mean"]:.4f} ± {rf_m["cv_f1_std"]:.4f}</td>'
                f'<td>{gb_m["cv_f1_mean"]:.4f} ± {gb_m["cv_f1_std"]:.4f}</td></tr>'
            )
            table_html = (
                '<table class="cmp-table">'
                '<thead><tr><th>Metric</th><th>Random Forest</th><th>Gradient Boosting</th></tr></thead>'
                '<tbody>' + "".join(rows_html) + cv_row + '</tbody>'
                '</table>'
            )

            best_badge = (
                f'<div class="best-badge" style="display:inline-flex;margin:0.75rem 0 0.25rem;">Best overall: {best}</div>'
                if best else ""
            )

            st.markdown(f"""
<div class="panel-card">
  <div class="panel-card-title">
    <div class="panel-card-icon">&#128200;</div>
    Head-to-Head Metrics
    <span style="margin-left:auto;font-size:0.7rem;color:#5a6a7d;font-weight:400;">Recall is most critical — a missed failure = unplanned downtime</span>
  </div>
  <div class="cmp-table-wrap">
  {table_html}
  </div>
  {best_badge}
</div>
""", unsafe_allow_html=True)

            cmp_data = {
                "Metric":           ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV F1 (mean ± std)"],
                "Random Forest":    [f"{rf_m['accuracy']:.4f}", f"{rf_m['precision']:.4f}", f"{rf_m['recall']:.4f}",
                                     f"{rf_m['f1']:.4f}", f"{rf_m['roc_auc']:.4f}",
                                     f"{rf_m['cv_f1_mean']:.4f} ± {rf_m['cv_f1_std']:.4f}"],
                "Gradient Boosting":[f"{gb_m['accuracy']:.4f}", f"{gb_m['precision']:.4f}", f"{gb_m['recall']:.4f}",
                                     f"{gb_m['f1']:.4f}", f"{gb_m['roc_auc']:.4f}",
                                     f"{gb_m['cv_f1_mean']:.4f} ± {gb_m['cv_f1_std']:.4f}"],
            }

            cats   = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
            rf_v   = [rf_m["accuracy"],rf_m["precision"],rf_m["recall"],rf_m["f1"],rf_m["roc_auc"]]
            gb_v   = [gb_m["accuracy"],gb_m["precision"],gb_m["recall"],gb_m["f1"],gb_m["roc_auc"]]

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("""
<div class="chart-label-card chart-label-blue">
  <span class="clc-dot"></span>Side-by-Side Metrics
</div>""", unsafe_allow_html=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(name="Random Forest",     x=cats, y=rf_v, marker_color="#3b82f6"))
                fig_bar.add_trace(go.Bar(name="Gradient Boosting", x=cats, y=gb_v, marker_color="#f59e0b"))
                fig_bar.add_hline(y=0.9, line_dash="dash", line_color="#16a34a", opacity=0.6,
                                  annotation_text="0.9 target", annotation_font_color="#16a34a")
                fig_bar.update_layout(**plotly_layout(barmode="group", title="",
                                                      yaxis=dict(range=[0,1.05]), height=300))
                st.plotly_chart(fig_bar, use_container_width=True)

            with bc2:
                st.markdown("""
<div class="chart-label-card chart-label-amber">
  <span class="clc-dot clc-dot-amber"></span>Performance Radar
</div>""", unsafe_allow_html=True)
                fig_rad = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(
                    r=rf_v+[rf_v[0]], theta=cats+[cats[0]],
                    fill="toself", name="Random Forest",
                    line_color="#3b82f6", fillcolor="rgba(59,130,246,0.15)"))
                fig_rad.add_trace(go.Scatterpolar(
                    r=gb_v+[gb_v[0]], theta=cats+[cats[0]],
                    fill="toself", name="Gradient Boosting",
                    line_color="#f59e0b", fillcolor="rgba(245,158,11,0.15)"))
                fig_rad.update_layout(**plotly_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1], color="#4a5568")),
                    title="", height=300))
                st.plotly_chart(fig_rad, use_container_width=True)

            if models_loaded:
                X_test, y_test = models["X_test"], models["y_test"]
                st.markdown("""
<div class="panel-card" style="margin-top:0.75rem;">
  <div class="panel-card-title">
    <div class="panel-card-icon">&#128200;</div>
    ROC Curves — Discrimination Ability
  </div>
  <p>The closer a curve hugs the top-left corner, the better the model separates failures from normal readings.</p>
</div>
""", unsafe_allow_html=True)
                fig_roc = go.Figure()
                for mk, lbl, col in [
                    ("random_forest",     "Random Forest",    "#3b82f6"),
                    ("gradient_boosting", "Gradient Boosting","#f59e0b"),
                ]:
                    y_prob = models[mk].predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc = models["metrics"][mk]["roc_auc"]
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{lbl} (AUC={auc:.3f})",
                                                 line=dict(color=col, width=2.5)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                             line=dict(color="#4a5568", width=1, dash="dash")))
                fig_roc.update_layout(**plotly_layout(
                    title="ROC Curves — closer to top-left is better",
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400))
                st.plotly_chart(fig_roc, use_container_width=True)

                st.markdown('<div class="sec-hdr">Confusion Matrices</div>', unsafe_allow_html=True)
                cm1, cm2 = st.columns(2)
                for col, mk, lbl in [(cm1,"random_forest","Random Forest"),(cm2,"gradient_boosting","Gradient Boosting")]:
                    with col:
                        y_pred = models[mk].predict(X_test)
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = px.imshow(
                            cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["No Failure","Failure"], y=["No Failure","Failure"],
                            color_continuous_scale=[[0,"#0f2240"],[1,"#3b82f6"]],
                            title=lbl,
                        )
                        fig_cm.update_layout(**plotly_layout(height=300))
                        st.plotly_chart(fig_cm, use_container_width=True)

    # ════════════════════════════════════════════════════
    # TAB 3 — FEATURE IMPACT
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
<div class="panel-card accent-purple" style="margin-bottom:1.25rem;">
  <div class="panel-card-title">
    <div class="panel-card-icon gi-purp">&#127989;</div>
    What Drives Failures? — Feature Impact
  </div>
  <p>Feature importance scores reveal which sensor readings the model relies on most when predicting failures. Higher-ranked features are the primary indicators of machine health.</p>
  <div class="info-pills">
    <span class="info-pill">Tool Wear</span>
    <span class="info-pill">Torque</span>
    <span class="info-pill">Power Proxy</span>
    <span class="info-pill">Temp Difference</span>
    <span class="info-pill">Rotational Speed</span>
    <span class="info-pill">Machine Type</span>
  </div>
</div>
""", unsafe_allow_html=True)

        if not metrics_available:
            st.markdown("""
<div class="panel-card accent-amber">
  <div class="panel-card-title">
    <div class="panel-card-icon gi-amber">&#9888;</div>
    Metrics Unavailable
  </div>
  <p>Feature importances require model metrics. Train models first.</p>
</div>
""", unsafe_allow_html=True)
        else:
            feat_labels = {
                "type_encoded":        "Machine Type (L/M/H)",
                "air_temperature":     "Air Temperature (K)",
                "process_temperature": "Process Temperature (K)",
                "rotational_speed":    "Rotational Speed (RPM)",
                "torque":              "Torque (Nm)",
                "tool_wear":           "Tool Wear (min)",
                "temp_diff":           "Temp Difference (\u0394)",
                "power_proxy":         "Power Proxy (T\u00d7RPM)",
            }
            st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)
            fi1, fi2 = st.columns(2, gap="medium")
            for col, key, lbl, hi, lo, card_cls, dot_cls in [
                (fi1, "rf_importances", "Random Forest",     "#3b82f6","#1e3a5f", "chart-label-blue",  ""),
                (fi2, "gb_importances", "Gradient Boosting", "#f59e0b","#3a2800", "chart-label-amber", "clc-dot-amber"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="chart-label-card {card_cls}">'
                        f'<span class="clc-dot {dot_cls}"></span>{lbl} — Feature Importances'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    imps = models["metrics"][key]
                    imp_df = (pd.DataFrame(list(imps.items()), columns=["Feature","Importance"])
                              .sort_values("Importance", ascending=True))
                    imp_df["Feature"] = imp_df["Feature"].map(feat_labels).fillna(imp_df["Feature"])
                    med = imp_df["Importance"].median()
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
                        marker_color=[hi if v > med else lo for v in imp_df["Importance"]],
                        text=imp_df["Importance"].round(4), textposition="outside",
                        textfont=dict(color="#8899aa", size=11),
                    ))
                    fig_imp.update_layout(**plotly_layout(
                        title="",
                        xaxis_title="Importance Score", height=380,
                        margin=dict(l=160, t=20, b=30, r=40),
                    ))
                    st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">How to Interpret These Scores</div>', unsafe_allow_html=True)

            _CELL = "background:#181b22;border:1px solid #22273a;border-radius:10px;padding:1rem 1.1rem;height:100%;"
            _NAME = "font-size:0.75rem;font-weight:700;color:#c8d8ea;margin-bottom:6px;"
            _DESC = "font-size:0.7rem;color:#5a6a7d;line-height:1.5;margin:0 0 8px;"
            _YES  = "display:inline-block;font-size:0.62rem;font-weight:700;padding:3px 9px;border-radius:8px;background:#0a2010;color:#4ade80;border:1px solid #166534;"
            _WARN = "display:inline-block;font-size:0.62rem;font-weight:700;padding:3px 9px;border-radius:8px;background:#1a1000;color:#fbbf24;border:1px solid #78350f;"

            features = [
                ("🔧", "Tool Wear",            "Minutes since the cutting tool was last replaced. As wear accumulates, surface quality degrades and mechanical stress rises.", "yes",  "✅ High value = elevated risk"),
                ("⚙️", "Torque (Nm)",          "Force applied during machining. Sustained high torque overloads the drive system, accelerating component fatigue.",            "yes",  "✅ High value = elevated risk"),
                ("⚡", "Power Proxy",           "Torque × RPM — a proxy for real power consumption. Sudden spikes signal mechanical stress events before sensors trigger alarms.", "yes", "✅ High value = elevated risk"),
                ("🌡️", "Temp Difference (Δ)",  "Process temperature minus air temperature. A large gap indicates poor heat dissipation — a precursor to thermal failure modes.", "yes",  "✅ High Δ = elevated risk"),
                ("🔄", "Rotational Speed",      "RPM of the machine spindle. Both extremes are risky — too high causes vibration; too low can stall bearings under load.",       "warn", "⚠️ Extremes are risky"),
                ("🏷️", "Machine Type (L/M/H)", "Quality grade of the machine (Low / Medium / High). Lower-grade machines have statistically higher baseline failure rates.",    "warn", "⚠️ Lower grade = higher rate"),
            ]

            # Render as two rows of 3 with explicit column gap
            for row_start in range(0, 6, 3):
                cols = st.columns(3, gap="medium")
                for col, (icon, name, desc, risk, badge) in zip(cols, features[row_start:row_start+3]):
                    badge_style = _YES if risk == "yes" else _WARN
                    with col:
                        st.markdown(
                            f'<div style="{_CELL}">'
                            f'<div style="{_NAME}">{icon} {name}</div>'
                            f'<div style="{_DESC}">{desc}</div>'
                            f'<span style="{badge_style}">{badge}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown('<div style="margin-bottom:0.75rem;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # TAB 4 — HOW IT WORKS
    # ════════════════════════════════════════════════════
    with tab4:
        # ── Row 1: Hero info cards ──────────────────────────
        st.markdown('<div class="sec-hdr">System Overview</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="hiw-hero">

  <div class="hiw-card">
    <div class="hiw-card-title">
      <div class="hiw-card-icon">🎯</div>
      What is Predictive Maintenance?
    </div>
    <p>Traditional maintenance falls into two costly traps — reactive or scheduled. Predictive maintenance is the smarter third path.</p>
    <div class="maint-row">
      <div class="maint-cell maint-reactive">
        <div class="maint-cell-label">Reactive</div>
        <p>Fix after it breaks. High downtime &amp; emergency costs.</p>
      </div>
      <div class="maint-cell maint-schedule">
        <div class="maint-cell-label">Scheduled</div>
        <p>Replace on a calendar. Wasteful — ignores actual wear.</p>
      </div>
      <div class="maint-cell maint-predict">
        <div class="maint-cell-label">Predictive ✓</div>
        <p>Act before failure. Right part, right time.</p>
      </div>
    </div>
    <p style="margin-top:0.8rem;">For manufacturers: <strong>30–50%</strong> less downtime &middot; <strong>10–40%</strong> lower costs &middot; longer asset life.</p>
  </div>

  <div class="hiw-card">
    <div class="hiw-card-title">
      <div class="hiw-card-icon">🤖</div>
      Why Machine Learning?
    </div>
    <p>Sensor data is complex. A machine fails due to a specific <em>combination</em> — high torque + worn tooling + heat buildup. No single sensor tells the whole story.</p>
    <p>Human-written rules can't capture interaction effects across 8 features simultaneously. ML learns the full pattern from historical failures and generalises automatically.</p>
    <div style="margin-top:0.8rem; padding:0.65rem 0.8rem; background:#0f1a2e; border-radius:10px; border:1px solid #1d3a5f;">
      <div style="font-size:0.68rem; font-weight:700; color:#60a5fa; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:5px;">📊 Dataset: AI4I 2020</div>
      <div class="ds-pills">
        <span class="ds-pill">10,000 readings</span>
        <span class="ds-pill">8 features</span>
        <span class="ds-pill">339 failures</span>
        <span class="ds-pill">5 failure modes</span>
        <span class="ds-pill">Binary classification</span>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

        # ── Row 2: Metrics + Workflow side by side ──────────
        m_col, w_col = st.columns(2)

        with m_col:
            st.markdown('<div class="sec-hdr">Key Metrics Explained</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="metrics-grid">

  <div class="metric-tile">
    <div class="metric-tile-icon mt-green">📈</div>
    <div class="metric-tile-body">
      <div class="metric-tile-name">Recall <span class="metric-tile-badge badge-critical">MOST CRITICAL</span></div>
      <div class="metric-tile-what">% of real failures your model caught</div>
      <div class="metric-tile-why">A missed failure = unplanned shutdown. Always optimise this first in maintenance contexts.</div>
    </div>
  </div>

  <div class="metric-tile">
    <div class="metric-tile-icon mt-blue">🎯</div>
    <div class="metric-tile-body">
      <div class="metric-tile-name">Precision<span class="metric-tile-badge badge-important">IMPORTANT</span></div>
      <div class="metric-tile-what">% of alerts that were real failures</div>
      <div class="metric-tile-why">Too low = alarm fatigue. Technicians start ignoring alerts.</div>
    </div>
  </div>

  <div class="metric-tile">
    <div class="metric-tile-icon mt-amber">⚖️</div>
    <div class="metric-tile-body">
      <div class="metric-tile-name">F1-Score</div>
      <div class="metric-tile-what">Harmonic mean of Recall + Precision</div>
      <div class="metric-tile-why">Best single metric to compare models when classes are imbalanced.</div>
    </div>
  </div>

  <div class="metric-tile">
    <div class="metric-tile-icon mt-purp">📐</div>
    <div class="metric-tile-body">
      <div class="metric-tile-name">ROC-AUC</div>
      <div class="metric-tile-what">Ranking ability across all thresholds</div>
      <div class="metric-tile-why">1.0 = perfect, 0.5 = random. Threshold-independent quality signal.</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

        with w_col:
            st.markdown('<div class="sec-hdr">Intelligence Workflow</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="wf-wrap">

  <div class="wf-step">
    <div class="wf-left"><div class="wf-num wf-num-blue">1</div><div class="wf-line"></div></div>
    <div class="wf-card">
      <div class="wf-card-title">📡 Data Acquisition<span class="wf-tag wf-tag-blue">INPUT</span></div>
      <div class="wf-card-body">Upload a CSV of machine sensor readings — temperature, RPM, torque, tool wear, and machine type.</div>
    </div>
  </div>

  <div class="wf-step">
    <div class="wf-left"><div class="wf-num wf-num-blue">2</div><div class="wf-line"></div></div>
    <div class="wf-card">
      <div class="wf-card-title">⚙️ Signal Processing<span class="wf-tag wf-tag-blue">TRANSFORM</span></div>
      <div class="wf-card-body">Columns normalised &middot; Units stripped &middot; Type encoded &middot; Temp &Delta; &amp; power proxy engineered &middot; StandardScaler applied.</div>
    </div>
  </div>

  <div class="wf-step">
    <div class="wf-left"><div class="wf-num wf-num-blue">3</div><div class="wf-line"></div></div>
    <div class="wf-card">
      <div class="wf-card-title">🧠 Predictive Analysis<span class="wf-tag wf-tag-blue">MODEL</span></div>
      <div class="wf-card-body">Random Forest or Gradient Boosting evaluates each row — returns binary prediction + failure probability (0–100%).</div>
    </div>
  </div>

  <div class="wf-step">
    <div class="wf-left"><div class="wf-num wf-num-blue">4</div><div class="wf-line"></div></div>
    <div class="wf-card">
      <div class="wf-card-title">🚨 Risk Classification<span class="wf-tag wf-tag-amb">SCORE</span></div>
      <div class="wf-card-body">Each machine assigned a tier based on confidence —</div>
      <div class="risk-pills">
        <span class="rp" style="background:#0a2010;color:#4ade80;border:1px solid #166534;">Low &lt;25%</span>
        <span class="rp" style="background:#1a1000;color:#fbbf24;border:1px solid #78350f;">Medium 25–60%</span>
        <span class="rp" style="background:#1e0808;color:#f87171;border:1px solid #991b1b;">High 60–85%</span>
        <span class="rp" style="background:#2d0505;color:#fca5a5;border:1px solid #991b1b;">Critical &gt;85%</span>
      </div>
    </div>
  </div>

  <div class="wf-step">
    <div class="wf-left"><div class="wf-num wf-num-green">5</div></div>
    <div class="wf-card" style="border-color:#166534;">
      <div class="wf-card-title">🛠️ Maintenance Intelligence<span class="wf-tag wf-tag-grn">OUTPUT</span></div>
      <div class="wf-card-body">Dashboard &middot; Fleet gauge &middot; Confidence distribution &middot; Feature drivers &middot; Exportable CSV for scheduling.</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid #22273a; color: #5a6a7d; font-size: 0.85rem; letter-spacing: 0.03em;">'
        'Predictive Maintenance Intelligence Platform &middot; Designed & Built By Tarun P'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()