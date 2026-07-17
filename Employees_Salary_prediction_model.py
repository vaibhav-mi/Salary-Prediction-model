import pandas as pd
import numpy as np
import datetime 
import streamlit as st
import joblib
from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401 (needed to unpickle the model)

MODEL_PATH = "employees_salary_predictor"

# ---------------------------------------------------------------------------
# Encoding maps -- must match exactly what the model was trained on.
# (Order of keys is the display order in each dropdown.)
# ---------------------------------------------------------------------------
EDUCATION_MAP = {"Diploma": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
GENDER_MAP = {"Female": 0, "Male": 1}
DEPARTMENT_MAP = {"Operations": 0, "IT": 1, "Finance": 2, "Sales": 3, "HR": 4, "Marketing": 5}
JOB_LEVEL_MAP = {"Junior": 0, "Mid": 1, "Senior": 2, "Lead": 3, "Manager": 4}
CITY_MAP = {"Hyderabad": 0, "Mumbai": 1, "Pune": 2, "Chennai": 3, "Bangalore": 4, "Delhi": 5}
REMOTE_MAP = {"No": 0, "Yes": 1}

FEATURE_ORDER = [
    "Age", "Gender", "Education", "Experience_Years", "Department", "Job_Level",
    "Performance_Rating", "Certifications", "Overtime_Hours", "Remote_Work",
    "City", "Company_Tenure", "Projects_Completed", "Skill_Score",
]

FEATURE_LABELS = {
    "Job_Level": "Job level", "Experience_Years": "Experience", "Education": "Education",
    "Skill_Score": "Skill score", "Performance_Rating": "Performance",
    "Projects_Completed": "Projects completed", "Department": "Department",
    "Certifications": "Certifications", "Age": "Age", "City": "City",
    "Company_Tenure": "Company tenure", "Overtime_Hours": "Overtime hours",
    "Remote_Work": "Remote work", "Gender": "Gender",
}

# A fixed mid-level profile used only as a comparison point, so the result can
# say "X% above/below a typical profile". It is scored against the live model
# on every run rather than hardcoded, so it always matches whatever model
# actually gets loaded.
BASELINE_PROFILE = {
    "Age": 29, "Gender": 0, "Education": 1, "Experience_Years": 5, "Department": 1,
    "Job_Level": 1, "Performance_Rating": 3, "Certifications": 2, "Overtime_Hours": 5,
    "Remote_Work": 1, "City": 4, "Company_Tenure": 3, "Projects_Completed": 8, "Skill_Score": 120,
}

st.set_page_config(page_title="Salary Ledger", page_icon="\U0001F4D2", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
baseline_prediction = model.predict(pd.DataFrame([BASELINE_PROFILE])[FEATURE_ORDER])[0]


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    [data-testid="stApp"] { background-color: #161B26; }
    .block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 4rem; }

    .ledger-eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8891A6; margin-bottom: 12px;
    }
    .ledger-title {
        font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 44px;
        color: #EDE7D9; margin: 0 0 12px 0; line-height: 1.15;
    }
    .ledger-subhead { color: #99A0B3; font-size: 15px; max-width: 580px; line-height: 1.6; }
    .ledger-rule { border: none; border-top: 0.5px solid #2E3648; margin: 2rem 0 1.75rem 0; }

    .section-head {
        font-family: 'IBM Plex Mono', monospace; font-size: 13px; letter-spacing: 1px;
        text-transform: uppercase; color: #C6CAD6; margin-bottom: 1.1rem;
    }
    .section-head .num { color: #C89B4A; margin-right: 8px; }

    [data-testid="stWidgetLabel"] p { color: #8891A6 !important; font-size: 13px !important; }

    .stButton button {
        background-color: #C89B4A; color: #161B26; border: none;
        font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 0.5px;
        border-radius: 6px; padding: 0.8rem 1rem; font-size: 14px; text-transform: uppercase;
    }
    .stButton button:hover { background-color: #DBAE5E; color: #161B26; }
    .stButton button:active { background-color: #B78A3B; }

    .result-card {
        background-color: #1F2635; border: 0.5px solid #2E3648; border-radius: 8px;
        padding: 30px 34px; margin-top: 1.75rem; animation: ledgerFadeIn 0.5s ease-out;
    }
    @keyframes ledgerFadeIn {
        from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8891A6; margin-bottom: 8px;
    }
    .result-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 52px; font-weight: 500;
        color: #EDE7D9; line-height: 1.1;
    }
    .result-delta { display: inline-block; font-size: 13px; color: #C89B4A; margin: 10px 0 26px 0; }
    .drivers-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 1.5px;
        text-transform: uppercase; color: #8891A6; margin-bottom: 16px;
        border-top: 0.5px solid #2E3648; padding-top: 20px;
    }
    .bar-row { display: flex; align-items: center; gap: 14px; margin-bottom: 11px; }
    .bar-label { width: 140px; flex-shrink: 0; font-size: 13px; color: #D8DCE5; }
    .bar-track { flex: 1; height: 6px; background: #2E3648; border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; background: #C89B4A; border-radius: 3px; }
    .bar-pct { width: 42px; text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8891A6; }

    .result-empty {
        border: 1px dashed #384157; border-radius: 8px; padding: 30px 34px;
        margin-top: 1.75rem; color: #7C8499; font-size: 14px; text-align: center;
    }

    .ledger-footer {
        margin-top: 3rem; padding-top: 1.5rem; border-top: 0.5px solid #2E3648;
        font-size: 12px; color: #6D7488;
    }
    .ledger-footer a { color: #C89B4A; text-decoration: none; }
    .ledger-footer a:hover { text-decoration: underline; }

    @media (max-width: 640px) {
        .ledger-title { font-size: 32px; }
        .result-value { font-size: 38px; }
    }
    @media (prefers-reduced-motion: reduce) {
        .result-card { animation: none; }
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()

st.markdown("""
<div class="ledger-eyebrow">Gradient boosting &middot; 14-factor compensation model</div>
<h1 class="ledger-title">Salary Ledger</h1>
<p class="ledger-subhead">Enter a profile below to get an instant, explainable salary estimate
&mdash; and exactly what drove the number.</p>
<hr class="ledger-rule">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 01 -- Profile
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head"><span class="num">01</span>Profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
age = c1.number_input("Age", min_value=17, max_value=80, value=29, step=1)
gender_label = c2.selectbox("Gender", list(GENDER_MAP.keys()), index=0)
education_label = c3.selectbox("Education level", list(EDUCATION_MAP.keys()), index=1)

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 02 -- Role & work
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head"><span class="num">02</span>Role &amp; work</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
department_label = c1.selectbox("Department", list(DEPARTMENT_MAP.keys()), index=1)
job_level_label = c2.selectbox("Job level", list(JOB_LEVEL_MAP.keys()), index=1)
experience = c3.number_input("Years of experience", min_value=0.0, max_value=70.0, value=5.0, step=1.0)
c4, c5 = st.columns(2)
city_label = c4.selectbox("City", list(CITY_MAP.keys()), index=4)
remote_label = c5.selectbox("Remote work", list(REMOTE_MAP.keys()), index=1)

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 03 -- Performance & skill
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head"><span class="num">03</span>Performance &amp; skill</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
performance = c1.slider("Performance rating", 0, 5, 3, step=1, help="Most recent appraisal rating, 0 to 5.")
certifications = c2.slider("Certifications earned", 0, 10, 2, step=1)
overtime = c3.number_input("Overtime hours / month", min_value=0.0, max_value=70.0, value=5.0, step=1.0)
c4, c5, c6 = st.columns(3)
tenure = c4.number_input("Company tenure (years)", min_value=0.0, max_value=15.0, value=3.0, step=1.0)
projects = c5.number_input("Projects completed", min_value=0.0, max_value=40.0, value=8.0, step=1.0)
skill_score = c6.number_input(
    "Skill score", min_value=50.0, max_value=200.0, value=120.0, step=1.0,
    help="Composite skill-assessment score from the training data, 50-200.",
)

st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
predict_clicked = st.button("Calculate my salary", width="stretch", type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    profile = {
        "Age": age, "Gender": GENDER_MAP[gender_label], "Education": EDUCATION_MAP[education_label],
        "Experience_Years": experience, "Department": DEPARTMENT_MAP[department_label],
        "Job_Level": JOB_LEVEL_MAP[job_level_label], "Performance_Rating": performance,
        "Certifications": certifications, "Overtime_Hours": overtime,
        "Remote_Work": REMOTE_MAP[remote_label], "City": CITY_MAP[city_label],
        "Company_Tenure": tenure, "Projects_Completed": projects, "Skill_Score": skill_score,
    }
    row = pd.DataFrame([profile])[FEATURE_ORDER]
    prediction = model.predict(row)[0]
    st.session_state["prediction"] = prediction
    st.session_state["delta_pct"] = (
        (prediction - baseline_prediction) / baseline_prediction * 100 if baseline_prediction else 0.0
    )

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    delta_pct = st.session_state["delta_pct"]
    if abs(delta_pct) < 0.5:
        delta_text = "&#8776; in line with a typical mid-level profile"
    else:
        arrow = "&#8593;" if delta_pct >= 0 else "&#8595;"
        direction = "above" if delta_pct >= 0 else "below"
        delta_text = f"{arrow} {abs(delta_pct):.1f}% {direction} a typical mid-level profile"

    pairs = sorted(zip(FEATURE_ORDER, model.feature_importances_), key=lambda p: -p[1])
    top, rest = pairs[:7], pairs[7:]
    rest_sum = sum(v for _, v in rest)

    bar_rows = "".join(
        f'<div class="bar-row"><div class="bar-label">{FEATURE_LABELS[name]}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{val * 100:.1f}%"></div></div>'
        f'<div class="bar-pct">{val * 100:.1f}%</div></div>'
        for name, val in top
    )
    if rest_sum > 0:
        bar_rows += (
            f'<div class="bar-row"><div class="bar-label">Other factors</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{rest_sum * 100:.1f}%"></div></div>'
            f'<div class="bar-pct">{rest_sum * 100:.1f}%</div></div>'
        )

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated annual salary (CTC)</div>
        <div class="result-value">&#8377;{prediction:.1f}L</div>
        <div class="result-delta">{delta_text}</div>
        <div class="drivers-label">What drove this number</div>
        {bar_rows}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="result-empty">Fill in the profile above and calculate to see your '
        "estimated salary and what drove it.</div>",
        unsafe_allow_html=True,
    )

with st.expander("How this model works"):
    st.markdown(
        "This estimate comes from a **Gradient Boosting Regressor** trained on 5000+ employee "
        "attributes. Job level and years of experience alone account for roughly 84% of the "
        "model's decision.\n\n"
        "This is a portfolio / educational project, not a certified compensation benchmark "
        "&mdash; treat the number as a data-driven estimate, not a guarantee."
    )

st.markdown("""
<div class="ledger-footer">
Built by Vaibhav &middot;
<a href="https://github.com/vaibhav-mi/Salary-Prediction-model" target="_blank">View source on GitHub</a>
</div>
""", unsafe_allow_html=True)
