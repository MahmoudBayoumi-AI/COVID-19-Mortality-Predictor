import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CoviCare — Bedside Risk Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model_pipelineE.pkl"

FEATURE_ORDER = [
    "medical_unit_level", "medical_unit", "gender", "care_type",
    "intubated", "pneumonia", "age", "pregnant", "diabetes", "copd",
    "asthma", "immunosuppressed", "high_pressure", "other_disease",
    "cardiovascular", "obesity", "chronic_renal", "tobacco_use",
    "covid_classification", "icu_admission",
]

# =========================================================
# DESIGN TOKENS
# ---------------------------------------------------------
# The look borrows from a bedside patient monitor rather than a
# generic dark dashboard: graphite screen, faint graticule grid,
# an amber "chart pen" as the working accent, and green / red
# reserved strictly for the clinical verdict (never for chrome) —
# the same discipline a triage board uses.
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    :root {
        --ink: #0a0d11;
        --panel: #10161d;
        --panel-2: #0c1218;
        --line: #1d2933;
        --amber: #f2a93c;
        --amber-dim: #8a662a;
        --cyan: #5fc9d8;
        --safe: #3ecf8e;
        --critical: #f0605a;
        --text: #e7edf0;
        --text-dim: #7c8f97;
        --grid: rgba(242,169,60,0.045);
    }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    code, .mono { font-family: 'IBM Plex Mono', monospace; }

    .stApp {
        background-color: var(--ink);
        background-image:
            repeating-linear-gradient(0deg, var(--grid) 0px, var(--grid) 1px, transparent 1px, transparent 28px),
            repeating-linear-gradient(90deg, var(--grid) 0px, var(--grid) 1px, transparent 1px, transparent 28px);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--panel-2);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { color: var(--text); }

    /* ---------- sidebar identity ---------- */
    .brand-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--amber); font-size: 0.68rem; letter-spacing: 3px; font-weight: 600;
    }
    .brand-title {
        font-size: 1.4rem; font-weight: 700; color: #ffffff; margin: 2px 0 2px 0; line-height: 1.2;
    }
    .brand-line { border: none; border-top: 1px dashed var(--line); margin: 14px 0 10px 0; }

    .nav-caption {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-dim); font-size: 0.65rem; letter-spacing: 2px;
        font-weight: 600; margin-top: 20px; margin-bottom: 8px; text-transform: uppercase;
    }

    /* segmented nav look for the radio group */
    div[data-testid="stRadio"] > div { gap: 4px; }
    div[data-testid="stRadio"] label {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 8px 10px !important;
        margin-bottom: 2px;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        border-color: var(--amber);
        background: rgba(242,169,60,0.08);
    }

    /* vitals readouts */
    .vital-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 14px; }
    .vital-num { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #ffffff; }
    .vital-label { font-size: 0.66rem; color: var(--text-dim); letter-spacing: 1px; text-transform: uppercase; text-align: right; max-width: 120px; }
    .vital-tick { height: 2px; background: linear-gradient(90deg, var(--amber) 0%, transparent 100%); margin-top: 4px; opacity: 0.5; }

    /* ---------- section headers ---------- */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--amber); font-size: 0.72rem; letter-spacing: 4px; font-weight: 600;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .section-head { font-size: 1.9rem; font-weight: 700; color: #ffffff; margin: 0 0 6px 0; }
    .section-sub { color: var(--text-dim); font-size: 0.95rem; margin-bottom: 4px; }

    /* ---------- ekg divider (signature element) ---------- */
    .ekg-wrap { margin: 14px 0 22px 0; }
    .ekg-path {
        stroke: var(--amber);
        stroke-width: 2;
        fill: none;
        stroke-dasharray: 600;
        stroke-dashoffset: 600;
        animation: draw 2.4s ease-out forwards;
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @media (prefers-reduced-motion: reduce) {
        .ekg-path { animation: none; stroke-dashoffset: 0; }
    }

    /* ---------- cards ---------- */
    .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
        transition: border-color 0.15s ease;
    }
    .card:hover { border-color: var(--amber-dim); }
    .card-tag {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--amber); font-weight: 600; font-size: 0.78rem;
        letter-spacing: 2px; margin-bottom: 12px; text-transform: uppercase;
        border-bottom: 1px solid var(--line); padding-bottom: 8px;
    }

    /* ---------- widgets ---------- */
    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label, div[data-testid="stRadio"] label {
        color: #a9bcc2 !important; font-weight: 500; font-size: 0.88rem;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    input[type="number"] {
        background-color: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div:focus-within, input[type="number"]:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 1px var(--amber) !important;
    }

    /* action button — amber ("chart pen"), never red: red is reserved for the verdict only */
    .stButton > button {
        background: var(--amber);
        color: #211404; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
        border: none; border-radius: 8px; padding: 14px 0; width: 100%; font-size: 0.9rem;
        transition: filter 0.15s ease, transform 0.15s ease;
    }
    .stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
    .stButton > button:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }

    /* ---------- verdict panel ---------- */
    .verdict {
        border-radius: 12px; padding: 26px 28px; display: flex; align-items: center; gap: 20px;
        border: 1px solid var(--line);
    }
    .verdict.safe { background: linear-gradient(90deg, rgba(62,207,142,0.10), transparent); border-left: 3px solid var(--safe); }
    .verdict.critical { background: linear-gradient(90deg, rgba(240,96,90,0.10), transparent); border-left: 3px solid var(--critical); }
    .pulse-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
    .pulse-dot.safe { background: var(--safe); box-shadow: 0 0 0 0 rgba(62,207,142,0.6); animation: pulse 1.6s infinite; }
    .pulse-dot.critical { background: var(--critical); box-shadow: 0 0 0 0 rgba(240,96,90,0.6); animation: pulse 1.6s infinite; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(240,96,90,0.55); }
        70% { box-shadow: 0 0 0 10px rgba(240,96,90,0); }
        100% { box-shadow: 0 0 0 0 rgba(240,96,90,0); }
    }
    @media (prefers-reduced-motion: reduce) { .pulse-dot { animation: none; } }
    .verdict-word { font-size: 1.7rem; font-weight: 700; margin: 0; line-height: 1.1; }
    .verdict-word.safe { color: var(--safe); }
    .verdict-word.critical { color: var(--critical); }
    .verdict-sub { color: var(--text-dim); font-size: 0.8rem; letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }

    /* risk meter */
    .meter-wrap { margin-top: 18px; }
    .meter-track {
        height: 8px; border-radius: 4px; width: 100%;
        background: linear-gradient(90deg, var(--safe) 0%, var(--amber) 55%, var(--critical) 100%);
        position: relative;
    }
    .meter-marker {
        position: absolute; top: -5px; width: 3px; height: 18px; background: #ffffff;
        border-radius: 2px; transform: translateX(-50%);
    }
    .meter-scale { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text-dim); margin-top: 6px; }

    hr { border-color: var(--line); }
    footer { visibility: hidden; }

    .foot-note {
        text-align: center; color: var(--text-dim); font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 1px; margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


def ekg_divider(height=54):
    """Signature element: a hand-drawn-feeling EKG trace, echoing a bedside monitor."""
    st.markdown(f"""
    <div class="ekg-wrap">
        <svg viewBox="0 0 600 60" width="100%" height="{height}" preserveAspectRatio="none">
            <path class="ekg-path" d="M0,30 L110,30 L130,30 L142,6 L156,54 L168,20 L180,30 L230,30
                     L340,30 L352,6 L366,54 L378,20 L390,30 L460,30 L600,30" />
        </svg>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

knn_pipe = load_model(MODEL_PATH)

# =========================================================
# SIDEBAR — IDENTITY, NAV, VITALS
# =========================================================
with st.sidebar:
    st.markdown('<div class="brand-eyebrow">BEDSIDE RISK MONITOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-title">🫀 CoviCare AI</div>', unsafe_allow_html=True)
    st.markdown('<hr class="brand-line">', unsafe_allow_html=True)

    st.markdown('<div class="nav-caption">CHART</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav", ["Predictor", "Model Insights", "About Dataset"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="nav-caption">Dataset Snapshot</div>', unsafe_allow_html=True)

    vitals = [
        ("1,025,152", "PATIENTS AFTER CLEANING"),
        ("20", "INPUT FEATURES"),
        ("13", "MEDICAL UNIT TYPES"),
        ("93.25%", "MODEL ACCURACY · KNN"),
    ]
    for num, label in vitals:
        st.markdown(f"""
        <div class="vital-row">
            <div>
                <div class="vital-num">{num}</div>
                <div class="vital-tick"></div>
            </div>
            <div class="vital-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="nav-caption">Active Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="mono" style="color:#fff;">KNeighborsClassifier</span><br>'
        '<span class="mono" style="color:var(--text-dim); font-size:0.8rem;">'
        'StandardScaler → KNN(k=5, distance)</span>',
        unsafe_allow_html=True,
    )

# =========================================================
# HELPERS
# =========================================================
YES_NO = {"No": 0, "Yes": 1}

def yes_no_select(label, key, default_index=0):
    choice = st.selectbox(label, list(YES_NO.keys()), index=default_index, key=key)
    return YES_NO[choice]

CLASSIFICATION_OPTIONS = {
    "1 — Positive, confirmed": 1,
    "2 — Positive, confirmed (contact-traced)": 2,
    "3 — Positive, confirmed (lab pending)": 3,
    "4 — Negative": 4,
    "5 — Inconclusive": 5,
    "6 — Not a carrier": 6,
    "7 — Awaiting result": 7,
}

# =========================================================
# PAGE: PREDICTOR
# =========================================================
if page == "Predictor":
    st.markdown('<div class="eyebrow">Risk Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">COVID-19 Mortality Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Chart the patient\'s clinical profile below to read an estimated outcome.</div>', unsafe_allow_html=True)
    ekg_divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-tag">Patient &amp; Care</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
        gender_label = st.selectbox("Gender", ["Female", "Male"])
        gender = 1 if gender_label == "Female" else 0

        care_label = st.selectbox("Care Type", ["Outpatient (returned home)", "Hospitalized"])
        care_type = 1 if care_label.startswith("Outpatient") else 0

        pregnant = yes_no_select("Pregnant", "pregnant")

        medical_unit_level = st.selectbox("Medical Unit Level", ["Level 1", "Level 2"])
        medical_unit_level = 1 if medical_unit_level == "Level 1" else 2

        medical_unit = st.selectbox("Medical Unit", list(range(1, 14)), index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-tag">COVID Status &amp; Critical Care</div>', unsafe_allow_html=True)
        classification_label = st.selectbox("COVID Classification", list(CLASSIFICATION_OPTIONS.keys()))
        covid_classification = CLASSIFICATION_OPTIONS[classification_label]
        pneumonia = yes_no_select("Pneumonia", "pneumonia")
        intubated = yes_no_select("Intubated", "intubated")
        icu_admission = yes_no_select("ICU Admission", "icu")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-tag">Pre-existing Conditions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            diabetes = yes_no_select("Diabetes", "diabetes")
            copd = yes_no_select("COPD", "copd")
            asthma = yes_no_select("Asthma", "asthma")
            immunosuppressed = yes_no_select("Immunosuppressed", "immuno")
            high_pressure = yes_no_select("High Blood Pressure", "hp")
        with c2:
            other_disease = yes_no_select("Other Disease", "other")
            cardiovascular = yes_no_select("Cardiovascular Disease", "cardio")
            obesity = yes_no_select("Obesity", "obesity")
            chronic_renal = yes_no_select("Chronic Renal Disease", "renal")
            tobacco_use = yes_no_select("Tobacco Use", "tobacco")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Read the Chart — Predict Outcome")

    if predict_clicked:
        if knn_pipe is None:
            st.error(
                f"Could not find **{MODEL_PATH}**. Place the pickled `knn_pipe` "
                "pipeline in the same folder as this app and reload."
            )
        else:
            row = {
                "medical_unit_level": medical_unit_level,
                "medical_unit": medical_unit,
                "gender": gender,
                "care_type": care_type,
                "intubated": intubated,
                "pneumonia": pneumonia,
                "age": age,
                "pregnant": pregnant,
                "diabetes": diabetes,
                "copd": copd,
                "asthma": asthma,
                "immunosuppressed": immunosuppressed,
                "high_pressure": high_pressure,
                "other_disease": other_disease,
                "cardiovascular": cardiovascular,
                "obesity": obesity,
                "chronic_renal": chronic_renal,
                "tobacco_use": tobacco_use,
                "covid_classification": covid_classification,
                "icu_admission": icu_admission,
            }
            input_df = pd.DataFrame([row], columns=FEATURE_ORDER)

            pred = knn_pipe.predict(input_df)[0]
            proba = knn_pipe.predict_proba(input_df)[0]
            labels = ["Alive", "Died"]
            verdict = labels[int(pred)]
            died_prob = proba[1] * 100
            alive_prob = proba[0] * 100
            state = "safe" if verdict == "Alive" else "critical"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="verdict {state}">
                <div class="pulse-dot {state}"></div>
                <div>
                    <div class="verdict-word {state}">{'ALIVE' if state=='safe' else 'DIED'}</div>
                    <div class="verdict-sub">PREDICTED OUTCOME</div>
                </div>
            </div>
            <div class="meter-wrap">
                <div class="meter-track">
                    <div class="meter-marker" style="left:{died_prob:.1f}%;"></div>
                </div>
                <div class="meter-scale">
                    <span>0% risk</span>
                    <span>50%</span>
                    <span>100% risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Mortality risk", f"{died_prob:.1f}%")
            m2.metric("Survival probability", f"{alive_prob:.1f}%")

            st.caption(
                "This tool provides a statistical estimate based on historical patterns and is "
                "**not** a clinical diagnosis. Always consult a medical professional."
            )

# =========================================================
# PAGE: MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":
    st.markdown('<div class="eyebrow">Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metrics reported on the held-out test set (20% split, SMOTE-balanced training data).</div>', unsafe_allow_html=True)
    ekg_divider(height=40)

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("93.25%", "ACCURACY"),
        ("53.17%", "PRECISION · DIED"),
        ("62.72%", "RECALL · DIED"),
        ("0.892", "ROC-AUC"),
    ]
    for col, (num, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="vital-num" style="font-size:1.6rem;">{num}</div>
                <div class="vital-label" style="text-align:center; max-width:none;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-tag">Pipeline</div>', unsafe_allow_html=True)
    st.code("knn_pipe = make_pipeline(\n    StandardScaler(),\n    KNeighborsClassifier(n_neighbors=5, weights='distance')\n)", language="python")
    st.write(
        "Trained on SMOTENC-resampled data to correct the strong class imbalance "
        "(≈93% Alive vs ≈7% Died in the raw data) before fitting."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-tag">Classification Report — Test Set</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame({
        "precision": [0.97, 0.53],
        "recall": [0.96, 0.63],
        "f1-score": [0.96, 0.58],
        "support": [189373, 14902],
    }, index=["Alive", "Died"])
    st.dataframe(report_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PAGE: ABOUT DATASET
# =========================================================
else:
    st.markdown('<div class="eyebrow">Source</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">About the Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Anonymized patient data released by the Mexican government, covering over 1 million COVID-19 cases with 21 original features.</div>', unsafe_allow_html=True)
    ekg_divider(height=40)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-tag">Feature Reference</div>', unsafe_allow_html=True)
    ref = pd.DataFrame({
        "Feature": [
            "medical_unit_level", "medical_unit", "gender", "care_type", "intubated",
            "pneumonia", "age", "pregnant", "diabetes", "copd", "asthma",
            "immunosuppressed", "high_pressure", "other_disease", "cardiovascular",
            "obesity", "chronic_renal", "tobacco_use", "covid_classification", "icu_admission",
        ],
        "Meaning": [
            "Level of the medical unit that treated the patient (1 or 2)",
            "ID of the specific health institution (1–13)",
            "1 = Female, 0 = Male",
            "1 = Outpatient (returned home), 0 = Hospitalized",
            "1 = Connected to ventilator, 0 = Not intubated",
            "1 = Has pneumonia / air-sac inflammation, 0 = No",
            "Patient age in years",
            "1 = Pregnant, 0 = Not pregnant",
            "1 = Has diabetes, 0 = No",
            "1 = Chronic obstructive pulmonary disease, 0 = No",
            "1 = Has asthma, 0 = No",
            "1 = Immunosuppressed, 0 = No",
            "1 = Has hypertension, 0 = No",
            "1 = Has another disease, 0 = No",
            "1 = Cardiovascular disease, 0 = No",
            "1 = Obese, 0 = No",
            "1 = Chronic renal disease, 0 = No",
            "1 = Tobacco user, 0 = No",
            "1–3 = Positive COVID (varying confirmation degree), 4–7 = Negative / inconclusive",
            "1 = Admitted to ICU, 0 = Not admitted",
        ],
    })
    st.dataframe(ref, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-tag">Target — DIED</div>', unsafe_allow_html=True)
    st.write(
        "Derived from the original `DATE_DIED` column: `9999-99-99` → **Alive (0)**, "
        "any real date → **Died (1)**. After cleaning missing values and dropping "
        "invalid ages, the final dataset used for training had **1,025,152** patients "
        "(before the age-0 filter) — roughly **93% Alive** vs **7% Died**, which is why "
        "SMOTE oversampling was applied before training."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="foot-note">COVICARE AI · MACHINE LEARNING COVID-19 MORTALITY PREDICTOR '
    '&nbsp;·&nbsp; PYTHON · SCIKIT-LEARN · STREAMLIT</div>',
    unsafe_allow_html=True,
)