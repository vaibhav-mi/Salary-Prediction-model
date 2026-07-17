import pandas as pd
import plotly.graph_objects as go
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

# ---------------------------------------------------------------------------
# Palette -- warm and bright on purpose. Cream/white instead of a dark
# background, coral as the energetic primary, teal for positive movement.
# ---------------------------------------------------------------------------
BG = "#FFF8F0"
CARD = "#FFFFFF"
BORDER = "#FFE0CC"
TEXT = "#3A2B27"
MUTED = "#8C7268"
CORAL = "#FF6B4A"
CORAL_DARK = "#E8532F"
TEAL = "#12A783"
TEAL_LIGHT = "#E3F5EF"
DIM_WARM = "#F4DFC9"

st.set_page_config(page_title="Payday - Salary Predictor", page_icon="\u2728", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
baseline_prediction = model.predict(pd.DataFrame([BASELINE_PROFILE])[FEATURE_ORDER])[0]


# ---------------------------------------------------------------------------
# Chart helpers -- every chart shares the same warm/coral styling so the
# native Streamlit widgets and the Plotly charts read as one interface.
# ---------------------------------------------------------------------------
def _style(fig, height, show_xaxis):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT, size=12),
        margin=dict(l=0, r=10, t=10, b=32 if show_xaxis else 10),
        height=height,
        showlegend=False,
        hoverlabel=dict(
            bgcolor=CARD, bordercolor=CORAL,
            font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT, size=12),
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=show_xaxis, color=MUTED)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


def driver_chart(pairs, rest_sum):
    labels = [FEATURE_LABELS[name] for name, _ in pairs]
    values = [v * 100 for _, v in pairs]
    if rest_sum > 0:
        labels.append("Other factors")
        values.append(rest_sum * 100)
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=CORAL),
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif", size=12),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(visible=True, color=TEXT, autorange="reversed", tickfont=dict(size=13))
    fig.update_xaxes(range=[0, max(values) * 1.3])
    return _style(fig, height=235, show_xaxis=False)


def job_level_chart(profile, current_label):
    names = list(JOB_LEVEL_MAP.keys())
    preds = []
    for name, val in JOB_LEVEL_MAP.items():
        trial = dict(profile)
        trial["Job_Level"] = val
        preds.append(model.predict(pd.DataFrame([trial])[FEATURE_ORDER])[0])
    colors = [CORAL if n == current_label else DIM_WARM for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=preds, marker=dict(color=colors),
        text=[f"\u20b9{v:.0f}L" for v in preds], textposition="outside",
        textfont=dict(color=TEXT, family="Plus Jakarta Sans, sans-serif", size=11),
        hovertemplate="%{x}: \u20b9%{y:.1f}L<extra></extra>",
    ))
    fig.update_xaxes(tickfont=dict(size=12))
    fig.update_yaxes(range=[0, max(preds) * 1.3])
    return _style(fig, height=210, show_xaxis=True)


def experience_chart(profile, current_exp, current_pred):
    exp_range = sorted(set(range(0, 31, 2)) | {int(round(current_exp))})
    preds = [
        model.predict(pd.DataFrame([{**profile, "Experience_Years": float(e)}])[FEATURE_ORDER])[0]
        for e in exp_range
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exp_range, y=preds, mode="lines", line=dict(color=TEAL, width=2.5),
        hovertemplate="%{x} yrs: \u20b9%{y:.1f}L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[current_exp], y=[current_pred], mode="markers",
        marker=dict(color=CORAL, size=10, line=dict(color="#FFFFFF", width=2)),
        hovertemplate="You: \u20b9%{y:.1f}L<extra></extra>",
    ))
    fig.update_xaxes(title=None, tickfont=dict(size=12))
    return _style(fig, height=210, show_xaxis=True)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    [data-testid="stApp"] {
        background: linear-gradient(180deg, #FFF8F0 0%, #FFF3E6 100%);
    }
    .block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 4rem; }

    .eyebrow-chip {
        display: inline-flex; align-items: center; gap: 6px;
        background: #FFEADD; color: #E8532F; font-weight: 600; font-size: 12.5px;
        padding: 6px 14px; border-radius: 999px; margin-bottom: 16px;
    }
    .ledger-title {
        font-family: 'Baloo 2', sans-serif; font-weight: 700; font-size: 48px;
        color: #FF6B4A; margin: 0 0 12px 0; line-height: 1.1;
    }
    .ledger-subhead { color: #8C7268; font-size: 16px; max-width: 560px; line-height: 1.6; }
    .ledger-rule { border: none; border-top: 1px solid #FFE0CC; margin: 2rem 0 1.75rem 0; }

    .section-head {
        display: flex; align-items: center; font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700; font-size: 17px; color: #3A2B27; margin-bottom: 1.1rem;
    }
    .num-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%; background: #FF6B4A;
        color: #FFFFFF; font-weight: 700; font-size: 13px; margin-right: 10px; flex-shrink: 0;
    }
    .section-note { color: #A0897D; font-size: 13px; margin: -0.6rem 0 1.1rem 0; }

    [data-testid="stWidgetLabel"] p { color: #8C7268 !important; font-size: 13px !important; font-weight: 500 !important; }

    div[data-baseweb="select"] > div, .stNumberInput input, .stSlider {
        border-color: #FFE0CC !important;
    }

    .live-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: #12A783; margin-right: 7px; animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .result-card {
        background: linear-gradient(135deg, #FFEEDF 0%, #FFE2CC 60%, #FFEAD8 100%);
        border: 1px solid #FFE0CC; border-radius: 16px 16px 0 0; border-bottom: none;
        padding: 32px 34px 22px 34px; margin-top: 1.75rem;
    }
    .result-label {
        font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12.5px; letter-spacing: 0.5px;
        text-transform: uppercase; color: #A0684E; margin-bottom: 8px; display: flex; align-items: center;
        font-weight: 600;
    }
    .result-value {
        font-family: 'Baloo 2', sans-serif; font-size: 56px; font-weight: 700;
        color: #E8532F; line-height: 1.1;
    }
    .result-delta {
        display: inline-block; font-size: 13px; font-weight: 600; margin: 14px 0 6px 0;
        padding: 6px 14px; border-radius: 999px;
    }
    .result-delta.positive { background: #E3F5EF; color: #0D8464; }
    .result-delta.neutral { background: #FFEADD; color: #B0603E; }
    .drivers-label {
        font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 13px;
        color: #3A2B27; margin-top: 16px; border-top: 1px solid #FFDBC4; padding-top: 18px;
    }

    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF; border: 1px solid #FFE0CC;
        border-radius: 12px; padding: 4px 18px 8px 18px;
    }
    .st-key-driver [data-testid="stPlotlyChart"] {
        border-radius: 0 0 16px 16px; border-top: none; margin-top: -1px;
    }
    .chart-title {
        font-size: 12.5px; font-weight: 600; letter-spacing: 0.3px; color: #A0897D; margin-bottom: 4px;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF; border: 1px solid #FFE0CC !important; border-radius: 12px;
    }

    .ledger-footer {
        margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #FFE0CC;
        font-size: 12.5px; color: #A0897D;
    }
    .ledger-footer a { color: #E8532F; text-decoration: none; font-weight: 600; }
    .ledger-footer a:hover { text-decoration: underline; }

    @media (max-width: 640px) {
        .ledger-title { font-size: 36px; }
        .result-value { font-size: 42px; }
    }
    @media (prefers-reduced-motion: reduce) {
        .live-dot { animation: none; }
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()

st.markdown("""
<div class="eyebrow-chip">\u2728 Gradient boosting &middot; 14-factor model</div>
<h1 class="ledger-title">Payday - Salary Predictor</h1>
<p class="ledger-subhead">Fill in a profile and watch your estimate come alive &mdash;
the number, its breakdown, and the scenarios all update instantly as you go.</p>
<hr class="ledger-rule">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 01 -- Profile
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head"><span class="num-badge">1</span>Profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
age = c1.number_input("Age", min_value=17, max_value=80, value=29, step=1)
gender_label = c2.selectbox("Gender", list(GENDER_MAP.keys()), index=0)
education_label = c3.selectbox("Education level", list(EDUCATION_MAP.keys()), index=1)

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 02 -- Role & work
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head"><span class="num-badge">2</span>Role &amp; work</div>', unsafe_allow_html=True)
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
st.markdown('<div class="section-head"><span class="num-badge">3</span>Performance &amp; skill</div>', unsafe_allow_html=True)
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

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live prediction -- recomputed on every rerun, i.e. every time any input
# above changes. No button: the whole point is that it reacts instantly.
# ---------------------------------------------------------------------------
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
delta_pct = (prediction - baseline_prediction) / baseline_prediction * 100 if baseline_prediction else 0.0

if abs(delta_pct) < 0.5:
    delta_text = "\u2248 in line with a typical mid-level profile"
    delta_class = "neutral"
else:
    arrow = "\u2191" if delta_pct >= 0 else "\u2193"
    direction = "above" if delta_pct >= 0 else "below"
    delta_text = f"{arrow} {abs(delta_pct):.1f}% {direction} a typical mid-level profile"
    delta_class = "positive" if delta_pct >= 0 else "neutral"

st.markdown(f"""
<div class="result-card">
    <div class="result-label"><span class="live-dot"></span>Estimated annual salary (CTC)</div>
    <div class="result-value">&#8377;{prediction:.1f}L</div>
    <div class="result-delta {delta_class}">{delta_text}</div>
    <div class="drivers-label">What drove this number</div>
</div>
""", unsafe_allow_html=True)

pairs = sorted(zip(FEATURE_ORDER, model.feature_importances_), key=lambda p: -p[1])
top, rest = pairs[:7], pairs[7:]
rest_sum = sum(v for _, v in rest)
with st.container(key="driver"):
    st.plotly_chart(driver_chart(top, rest_sum), width="stretch", theme=None, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Section 04 -- Explore. Two live "what if" scenarios built on the current
# profile: the two features that account for ~84% of the model's decision.
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head" style="margin-top: 2.25rem;"><span class="num-badge">4</span>Explore</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Same profile as above, one factor varied at a time. '
    "Hover a point for the exact number.</div>",
    unsafe_allow_html=True,
)
ec1, ec2 = st.columns(2)
with ec1:
    st.markdown('<div class="chart-title">SALARY BY JOB LEVEL</div>', unsafe_allow_html=True)
    st.plotly_chart(job_level_chart(profile, job_level_label), width="stretch", theme=None, config={"displayModeBar": False})
with ec2:
    st.markdown('<div class="chart-title">SALARY BY YEARS OF EXPERIENCE</div>', unsafe_allow_html=True)
    st.plotly_chart(experience_chart(profile, experience, prediction), width="stretch", theme=None, config={"displayModeBar": False})

with st.expander("How this model works"):
    st.markdown(
        "This estimate comes from a **Gradient Boosting Regressor** trained on 14 employee "
        "attributes. Job level and years of experience alone account for roughly 84% of the "
        "model's decision -- which is why they're the two factors broken out above.\n\n"
        "This is a portfolio / educational project, not a certified compensation benchmark "
        "&mdash; treat the number as a data-driven estimate, not a guarantee."
    )

st.markdown("""
<div class="ledger-footer">
Built by Vaibhav &middot;
<a href="https://github.com/vaibhav-mi/Salary-Prediction-model" target="_blank">View source on GitHub</a>
</div>
""", unsafe_allow_html=True)
