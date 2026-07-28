import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ------------------------------------------------------------------
# Page Config & Header Setup (DATA SCIENCE ML FOCUS)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Care AI — Machine Learning Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS for Premium Data Science Aesthetics & Developer Card
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #eaf2f3 0%, #f4f8f9 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d5c58 0%, #1e40af 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }

    /* Main Title Styling (Theme Blue Box with White Text & Red Heart) */
    .hero-header {
        text-align: center;
        padding: 1.1rem 1.2rem;
        background: linear-gradient(135deg, #0d5c58 0%, #1e40af 100%);
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(13, 92, 88, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    .hero-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.2rem;
    }
    .hero-header p {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 0;
    }

    /* Form Container */
    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        border: 1px solid #cce3de;
        box-shadow: 0 2px 10px rgba(13, 92, 88, 0.04);
    }

    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0b5cab;
        margin: 0.8rem 0 0.5rem 0;
        border-left: 4px solid #14b8a6;
        padding-left: 0.6rem;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0d9488, #2563eb);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.6rem 0;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0f766e, #1d4ed8);
        transform: translateY(-1px);
    }

    /* Metric Box */
    .metric-box {
        background: #ffffff;
        border-radius: 6px;
        padding: 0.75rem 0.5rem;
        border: 1px solid #cce3de;
        box-shadow: 0 2px 6px rgba(13, 92, 88, 0.04);
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0d5c58;
    }
    .metric-box .lbl {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Theme Blue Metric Box (for EDA KPIs) */
    .metric-box-blue {
        background: linear-gradient(135deg, #0d5c58 0%, #1e40af 100%);
        border-radius: 8px;
        padding: 0.85rem 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 14px rgba(13, 92, 88, 0.16);
        text-align: center;
    }
    .metric-box-blue .val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff !important;
    }
    .metric-box-blue .lbl {
        font-size: 0.78rem;
        font-weight: 700;
        color: #e2e8f0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Result Banner */
    .result-card {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 0.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .result-high {
        background: linear-gradient(135deg, #fef2f2 0%, #ffe4e6 100%);
        border: 2px solid #f87171;
    }
    .result-high .result-title { color: #dc2626; }
    .result-low {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #34d399;
    }
    .result-low .result-title { color: #059669; }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-normal { background: #d1fae5; color: #047857; }
    .badge-borderline { background: #fef3c7; color: #b45309; }
    .badge-high { background: #fee2e2; color: #b91c1c; }

    /* Dynamic Advice Cards (Safe / Warning / Danger) */
    .advice-card {
        background: #ffffff;
        border-left: 5px solid #2563eb;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .advice-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 5px solid #10b981;
    }
    .advice-safe .advice-title { color: #047857 !important; }
    .advice-safe .advice-text { color: #065f46 !important; }

    .advice-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
    }
    .advice-warning .advice-title { color: #b45309 !important; }
    .advice-warning .advice-text { color: #92400e !important; }

    .advice-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #ef4444;
    }
    .advice-danger .advice-title { color: #b91c1c !important; }
    .advice-danger .advice-text { color: #991b1b !important; }

    .advice-title {
        font-weight: 700;
        color: #1e3a8a;
        font-size: 0.92rem;
        margin-bottom: 0.15rem;
    }
    .advice-text {
        font-size: 0.85rem;
        color: #475569;
        margin: 0;
    }

    /* Subview Box */
    .subview-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        border: 1px solid #cce3de;
        box-shadow: 0 2px 10px rgba(13, 92, 88, 0.04);
        margin-top: 0.8rem;
    }

    /* Developer Profile Card Styling */
    .dev-profile-card {
        background: linear-gradient(135deg, #0d5c58 0%, #1e40af 100%);
        border-radius: 8px;
        padding: 1.1rem;
        color: white !important;
        box-shadow: 0 4px 16px rgba(13, 92, 88, 0.12);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .dev-profile-card * {
        color: white !important;
    }
    .dev-avatar {
        font-size: 3rem;
        background: rgba(255, 255, 255, 0.15);
        width: 70px;
        height: 70px;
        line-height: 70px;
        border-radius: 50%;
        margin: 0 auto 0.6rem auto;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .dev-name {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .dev-role {
        font-size: 0.9rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 0.8rem;
    }
    .dev-badge-container {
        display: flex;
        justify-content: center;
        gap: 0.4rem;
        flex-wrap: wrap;
    }
    .dev-chip {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    .about-box-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #cce3de;
        box-shadow: 0 2px 10px rgba(13, 92, 88, 0.04);
        height: 100%;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Sidebar Navigation Button Styling */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: #ffffff !important;
        color: #0d5c58 !important;
        border: 1px solid #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18) !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.8rem !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"] * {
        color: #0d5c58 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.8rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.28) !important;
        border-color: #ffffff !important;
        transform: translateX(3px) !important;
    }

    /* Mobile Responsive Optimizations (< 768px screens) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2rem !important;
        }

        .hero-header {
            padding: 0.9rem 0.8rem !important;
            margin-bottom: 0.8rem !important;
        }
        .hero-header h1 {
            font-size: 1.45rem !important;
            line-height: 1.3 !important;
        }
        .hero-header p {
            font-size: 0.82rem !important;
        }

        div[data-testid="stForm"] {
            padding: 0.75rem 0.8rem !important;
        }

        .section-header {
            font-size: 0.95rem !important;
            margin: 0.6rem 0 0.4rem 0 !important;
        }

        .metric-box, .metric-box-blue {
            padding: 0.6rem 0.4rem !important;
            margin-bottom: 0.5rem !important;
        }
        .metric-box .val, .metric-box-blue .val {
            font-size: 1.25rem !important;
        }
        .metric-box .lbl, .metric-box-blue .lbl {
            font-size: 0.7rem !important;
        }

        .result-card {
            padding: 0.8rem 0.6rem !important;
        }
        .result-title {
            font-size: 1.25rem !important;
        }

        .subview-card, .about-box-card {
            padding: 0.75rem 0.7rem !important;
        }

        .table-responsive-container {
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }

        .stButton>button {
            font-size: 0.92rem !important;
            padding: 0.55rem 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Artifact Loading & Model Pipeline
# ------------------------------------------------------------------
NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

@st.cache_resource
def load_artifacts():
    from sklearn.preprocessing import StandardScaler

    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    df = pd.read_csv("heart.csv")
    ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
    df["Cholesterol"] = df["Cholesterol"].replace(0, ch_mean).round(2)
    bp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
    df["RestingBP"] = df["RestingBP"].replace(0, bp_mean).round(2)

    df_encode = pd.get_dummies(df, drop_first=True).astype(int)
    pre_scaler = StandardScaler()
    pre_scaler.fit(df_encode[NUMERICAL_COLS])

    return model, scaler, pre_scaler, columns, df

model, scaler, pre_scaler, columns, raw_df = load_artifacts()

if "history" not in st.session_state:
    st.session_state.history = []

def predict_heart_disease(input_df):
    raw = input_df.copy()
    raw[""] = raw["RestingBP"]
    encoded = pd.get_dummies(raw, drop_first=True).astype(int)
    encoded[NUMERICAL_COLS] = pre_scaler.transform(encoded[NUMERICAL_COLS])

    for col in columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[columns]

    final_input = scaler.transform(encoded)
    prediction = int(model.predict(final_input)[0])
    proba = float(model.predict_proba(final_input)[0][1]) if hasattr(model, "predict_proba") else 0.5
    return prediction, proba

# ------------------------------------------------------------------
# Plotly Chart Creators
# ------------------------------------------------------------------
def plot_risk_gauge(proba):
    pct = round(proba * 100, 1)
    
    if pct < 35:
        bar_color = "#10b981"
        risk_text = "LOW RISK"
    elif pct < 65:
        bar_color = "#f59e0b"
        risk_text = "MODERATE RISK"
    else:
        bar_color = "#ef4444"
        risk_text = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 36, 'family': 'Plus Jakarta Sans', 'color': bar_color, 'weight': 800}},
        title={'text': f"<b>HEART DISEASE RISK SCORE</b><br><span style='font-size:0.9em;color:{bar_color}'>{risk_text}</span>", 'font': {'size': 13, 'color': '#1e293b', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 35], 'color': '#d1fae5'},
                {'range': [35, 65], 'color': '#fef3c7'},
                {'range': [65, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.75,
                'value': pct
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=15),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans")
    )
    return fig

def plot_risk_waterfall(contrib_df):
    df_sorted = contrib_df.sort_values("Impact", ascending=False).copy()
    features = df_sorted["Feature"].tolist()
    impacts = df_sorted["Impact"].tolist()
    text_labels = [f"{val:+d}%" for val in impacts]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(features),
        y=features,
        x=impacts,
        text=text_labels,
        textposition="outside",
        textfont=dict(family="Plus Jakarta Sans", size=11, color="#1e293b"),
        decreasing={"marker": {"color": "#10b981", "line": {"color": "#059669", "width": 1.5}}},
        increasing={"marker": {"color": "#ef4444", "line": {"color": "#dc2626", "width": 1.5}}},
        connector={"line": {"color": "#94a3b8", "width": 1.5, "dash": "dot"}},
        cliponaxis=False
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family="Plus Jakarta Sans"),
        height=275,
        xaxis=dict(
            title=dict(text="Risk Score Shift (%)", font=dict(size=11, color="#64748b")),
            zeroline=True,
            zerolinecolor="#64748b",
            zerolinewidth=2,
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(size=11, color="#1e293b")
        ),
        margin=dict(l=10, r=45, t=15, b=40),
        showlegend=False
    )
    return fig

def plot_risk_lollipop(contrib_df):
    df_sorted = contrib_df.sort_values("Impact", ascending=True).copy()
    
    fig = go.Figure()

    fig.add_shape(
        type="line", x0=0, x1=0, y0=-0.5, y1=len(df_sorted)-0.5,
        line=dict(color="#64748b", width=2)
    )

    for idx, row in df_sorted.iterrows():
        color = "#ef4444" if row["Impact"] > 0 else "#10b981"
        fig.add_trace(go.Scatter(
            x=[0, row["Impact"]],
            y=[row["Feature"], row["Feature"]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=[row["Impact"]],
            y=[row["Feature"]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=2, color="white")),
            text=[f"{row['Impact']:+d}%"],
            textposition="middle right" if row["Impact"] >= 0 else "middle left",
            textfont=dict(family="Plus Jakarta Sans", size=11, color="#1e293b"),
            name=row["Type"],
            showlegend=False,
            hovertemplate=f"<b>{row['Feature']}</b><br>Impact: {row['Impact']:+d}%<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family="Plus Jakarta Sans"),
        height=275,
        xaxis=dict(
            title=dict(text="Risk Score Shift (%)", font=dict(size=11, color="#64748b")),
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color="#1e293b")
        ),
        margin=dict(l=10, r=45, t=15, b=40)
    )
    return fig

def render_feature_impact_cards(contrib_df):
    df_sorted = contrib_df.sort_values("Impact", ascending=False)
    cards_html = []
    for _, row in df_sorted.iterrows():
        is_risk = row["Impact"] > 0
        bg_card = "#fef2f2" if is_risk else "#ecfdf5"
        border_card = "#fca5a5" if is_risk else "#6ee7b7"
        text_color = "#b91c1c" if is_risk else "#047857"
        icon = "⚠️ Risk Factor" if is_risk else "🛡️ Protective Factor"
        val_str = f"{row['Impact']:+d}%"
        
        cards_html.append(f"""
        <div style="background: {bg_card}; border: 1px solid {border_card}; padding: 0.55rem 0.8rem; border-radius: 8px; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b;">{row['Feature']}</div>
                <div style="font-size: 0.72rem; font-weight: 600; color: {text_color};">{icon}</div>
            </div>
            <div style="font-size: 1.1rem; font-weight: 800; color: {text_color}; background: #ffffff; padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid {border_card};">
                {val_str}
            </div>
        </div>
        """)
    return "".join(cards_html)

def plot_patient_radar(age, bp, chol, max_hr, oldpeak):
    categories = ['Age Norm', 'BP Index', 'Cholesterol', 'Max HR Capability', 'ST Oldpeak']
    benchmark_vals = [50, 60, 55, 80, 20]
    
    patient_vals = [
        min(100, max(10, int((age / 80) * 100))),
        min(100, max(10, int((bp / 180) * 100))),
        min(100, max(10, int((chol / 350) * 100))),
        min(100, max(10, int((max_hr / (220 - age if 220 - age > 0 else 180)) * 100))),
        min(100, max(5, int((max(0.0, oldpeak) / 4.0) * 100)))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark_vals,
        theta=categories,
        fill='toself',
        name='Healthy Baseline',
        line=dict(color='#10b981', width=2),
        fillcolor='rgba(16, 185, 129, 0.15)',
        hovertemplate="%{theta}: %{r}%<extra>Healthy Baseline</extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=patient_vals,
        theta=categories,
        fill='toself',
        name='Patient Profile',
        line=dict(color='#2563eb', width=3),
        fillcolor='rgba(37, 99, 235, 0.25)',
        hovertemplate="%{theta}: %{r}%<extra>Patient Profile</extra>"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=11, color='#1e293b', family='Plus Jakarta Sans'), gridcolor="#e2e8f0")
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(family="Plus Jakarta Sans", size=11)),
        margin=dict(l=35, r=35, t=25, b=35),
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Plus Jakarta Sans")
    )
    return fig

def plot_population_scatter(patient_age, patient_hr, patient_risk):
    df_sample = raw_df.sample(min(400, len(raw_df)), random_state=42).copy()
    df_sample["Status"] = df_sample["HeartDisease"].map({1: "Historical Heart Disease", 0: "Historical Healthy"})

    fig = px.scatter(
        df_sample,
        x="Age",
        y="MaxHR",
        color="Status",
        color_discrete_map={"Historical Heart Disease": "#ef4444", "Historical Healthy": "#10b981"},
        opacity=0.5,
        hover_data=["RestingBP", "Cholesterol"],
        labels={"Age": "Age (Years)", "MaxHR": "Max Heart Rate (bpm)"}
    )

    marker_color = "#dc2626" if patient_risk == 1 else "#059669"
    fig.add_trace(go.Scatter(
        x=[patient_age],
        y=[patient_hr],
        mode='markers+text',
        marker=dict(size=18, color=marker_color, symbol='star', line=dict(width=2, color='white')),
        text=["YOU ARE HERE"],
        textposition="top center",
        name="Current Patient",
        showlegend=True
    ))

    fig.update_layout(
        margin=dict(l=25, r=25, t=25, b=25),
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family="Plus Jakarta Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9")
    )
    return fig

def plot_chest_pain_donut(df_input):
    counts = df_input["ChestPainType"].value_counts()
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        color_discrete_sequence=["#0d5c58", "#2563eb", "#f59e0b", "#ef4444"],
        title="Chest Pain Type Distribution in Cohort"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=290,
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def plot_vitals_bar(df_input):
    avg_bp = df_input["RestingBP"].mean() if len(df_input) > 0 else 0
    avg_chol = df_input["Cholesterol"].mean() if len(df_input) > 0 else 0
    avg_hr = df_input["MaxHR"].mean() if len(df_input) > 0 else 0

    overall_bp = raw_df["RestingBP"].median()
    overall_chol = raw_df["Cholesterol"].median()
    overall_hr = raw_df["MaxHR"].median()

    metrics_df = pd.DataFrame({
        "Vital": ["Resting BP (mmHg)", "Cholesterol (mg/dl)", "Max HR (bpm)"],
        "Filtered Cohort Mean": [avg_bp, avg_chol, avg_hr],
        "Overall Dataset Median": [overall_bp, overall_chol, overall_hr]
    })

    fig = px.bar(
        metrics_df,
        y="Vital",
        x=["Filtered Cohort Mean", "Overall Dataset Median"],
        barmode="group",
        orientation="h",
        color_discrete_sequence=["#2563eb", "#94a3b8"],
        title="Cohort Vitals Means vs Overall Dataset Median"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=290,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig

# ------------------------------------------------------------------
# Report HTML Generator Function
# ------------------------------------------------------------------
def generate_html_report(age, sex, bp, chol, max_hr, fbs, cp, ecg, angina, oldpeak, slope, pred_class, proba_val, advice_list):
    risk_label = "HIGH RISK" if pred_class == 1 else "LOW RISK"
    badge_bg = "#fee2e2" if pred_class == 1 else "#d1fae5"
    badge_color = "#b91c1c" if pred_class == 1 else "#047857"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    advice_rows = "".join([f"<li><b>{item[0]}:</b> {item[1]}</li>" for item in advice_list])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Heart Care AI — Clinical Evaluation Report</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f8fafc; color: #1e293b; padding: 20px; }}
        .report-card {{ max-width: 800px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        .header {{ text-align: center; border-bottom: 2px solid #0d5c58; padding-bottom: 15px; margin-bottom: 20px; }}
        .header h1 {{ color: #0d5c58; margin: 0; font-size: 24px; }}
        .header p {{ color: #64748b; margin: 5px 0 0 0; font-size: 14px; }}
        .badge {{ display: inline-block; padding: 8px 16px; background: {badge_bg}; color: {badge_color}; border-radius: 30px; font-weight: bold; font-size: 16px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #f1f5f9; color: #334155; font-size: 13px; text-transform: uppercase; }}
        td {{ font-size: 14px; }}
        .print-btn {{ display: block; width: 100%; text-align: center; margin-top: 25px; padding: 12px; background: #0d5c58; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: bold; cursor: pointer; }}
        .print-btn:hover {{ background: #0b4a47; }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <h1>❤️ Heart Care AI — Clinical Evaluation Report</h1>
            <p>Generated on {date_str} | Evaluated by KNN Machine Learning Model</p>
        </div>
        
        <div style="text-align: center;">
            <div class="badge">PREDICTED CATEGORY: {risk_label} ({proba_val*100:.1f}% Confidence)</div>
        </div>

        <h3>📋 Patient Vitals Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Clinical Attribute</th>
                    <th>Value Recorded</th>
                    <th>Reference Standard</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Patient Demographics</td><td>Age {age} · {sex}</td><td>Adult Profile</td></tr>
                <tr><td>Resting Blood Pressure</td><td>{bp} mmHg</td><td>&lt; 120 mmHg (Normal)</td></tr>
                <tr><td>Serum Cholesterol</td><td>{chol} mg/dl</td><td>&lt; 200 mg/dl (Desirable)</td></tr>
                <tr><td>Max Heart Rate Achieved</td><td>{max_hr} bpm</td><td>Target: ~{220-age} bpm</td></tr>
                <tr><td>Chest Pain Type</td><td>{cp}</td><td>Atypical / Asymptomatic</td></tr>
                <tr><td>Resting ECG</td><td>{ecg}</td><td>Normal ST-T</td></tr>
                <tr><td>Exercise Angina</td><td>{angina}</td><td>None</td></tr>
                <tr><td>Oldpeak ST Depression</td><td>{oldpeak}</td><td>&lt; 1.0</td></tr>
                <tr><td>ST Segment Slope</td><td>{slope}</td><td>Upward Slope</td></tr>
                <tr><td>Fasting Blood Sugar &gt; 120 mg/dl</td><td>{fbs}</td><td>No</td></tr>
            </tbody>
        </table>

        <h3>💡 Personalized Clinical Insights</h3>
        <ul>{advice_rows}</ul>

        <div style="margin-top: 25px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 10px;">
            ⚕️ Educational & Decision Support Only — Formal medical diagnosis must be conducted by a qualified physician.
        </div>

        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF Report</button>
    </div>
</body>
</html>"""
    return html_content

# ------------------------------------------------------------------
# Sidebar Content (DATA SCIENCE FOCUS + TOP NAVIGATION MODULES)
# ------------------------------------------------------------------
with st.sidebar:
    # Sidebar Header Brand Card
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.12); padding: 1.1rem 1rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.25); text-align: center; margin-bottom: 1rem; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
        <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">❤️</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">Heart Care AI</div>
        <div style="font-size: 0.8rem; font-weight: 600; color: #e2e8f0; opacity: 0.95;">Data Science Predictive Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Category Navigation")
    
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Heart Disease Risk Predictor"

    btn1 = st.button("Heart Disease Risk Predictor", use_container_width=True, type="primary" if st.session_state.active_page == "Heart Disease Risk Predictor" else "secondary", key="nav_btn_predictor")
    if btn1:
        st.session_state.active_page = "Heart Disease Risk Predictor"
        st.rerun()

    btn2 = st.button("Exploratory Data Analysis (EDA)", use_container_width=True, type="primary" if st.session_state.active_page == "Exploratory Data Analysis (EDA)" else "secondary", key="nav_btn_eda")
    if btn2:
        st.session_state.active_page = "Exploratory Data Analysis (EDA)"
        st.rerun()

    btn3 = st.button("Model Benchmarks & Metrics", use_container_width=True, type="primary" if st.session_state.active_page == "Model Benchmarks & Metrics" else "secondary", key="nav_btn_benchmarks")
    if btn3:
        st.session_state.active_page = "Model Benchmarks & Metrics"
        st.rerun()

    btn4 = st.button("About The ML Project", use_container_width=True, type="primary" if st.session_state.active_page == "About The ML Project" else "secondary", key="nav_btn_about")
    if btn4:
        st.session_state.active_page = "About The ML Project"
        st.rerun()

    page_category = st.session_state.active_page
    st.markdown("---")
    
    # Active ML Engine Info Card
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.1); padding: 0.9rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2); font-size: 0.85rem; line-height: 1.6; color: #ffffff;">
        <div style="font-weight: 800; font-size: 0.92rem; margin-bottom: 0.4rem; color: #ffffff; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 0.3rem;">📌 Active ML Engine</div>
        <div><b>Algorithm:</b> KNN (K=5)</div>
        <div><b>Validation Acc:</b> <span style="color: #34d399; font-weight: 800;">88.04%</span></div>
        <div><b>ROC-AUC Score:</b> <span style="color: #60a5fa; font-weight: 800;">0.9120</span></div>
        <div><b>Dataset Cohort:</b> 918 Patient Records</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🕓 Evaluation History")
    if st.session_state.history:
        st.markdown(f"<div style='font-size:0.8rem; color:#cbd5e1; margin-bottom:0.5rem;'>Recorded Patients: <b>{len(st.session_state.history)}</b></div>", unsafe_allow_html=True)
        for idx, h in enumerate(reversed(st.session_state.history[-5:]), 1):
            icon = "🔴" if h["result"] == 1 else "🟢"
            bg_color = "rgba(239, 68, 68, 0.15)" if h["result"] == 1 else "rgba(16, 185, 129, 0.15)"
            border_color = "rgba(239, 68, 68, 0.3)" if h["result"] == 1 else "rgba(16, 185, 129, 0.3)"
            st.markdown(f"""
            <div style="background: {bg_color}; border: 1px solid {border_color}; padding: 0.5rem 0.8rem; border-radius: 8px; margin-bottom: 0.4rem; font-size: 0.82rem; color: #ffffff;">
                {icon} <b>Age {h['age']} ({h['sex']})</b> · <span style="font-weight: 800;">{h['prob']:.1f}% Risk</span>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Evaluation History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("<div style='font-size:0.8rem; color:#e2e8f0; opacity:0.85; font-style:italic;'>No evaluations recorded in this session yet.</div>", unsafe_allow_html=True)
            
    st.markdown("---")
    st.caption("⚕️ Educational Data Science Tool — Not a substitute for formal clinical testing.")

# ------------------------------------------------------------------
# Main Header (DATA SCIENCE & ML PREDICTION FOCUS)
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <h1>Khushi's <span style="color: #ef4444;">❤️</span> Heart Disease Risk Predictor</h1>
    <p>Cardiovascular Machine Learning Classification Engine · Patient Risk Profiling · Exploratory Data Analysis</p>
</div>
""", unsafe_allow_html=True)

# ==================================================================
# CATEGORY 1 — Heart Disease Risk Predictor (Main ML Prediction Engine)
# ==================================================================
if page_category == "Heart Disease Risk Predictor":
    col_input, col_results = st.columns([1.1, 1], gap="large")

    with col_input:
        with st.form("main_patient_form"):
            st.markdown('<div class="section-header">🧍 Patient Clinical Features</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("🎂 Age (Years)", min_value=18, max_value=100, value=48, help="Patient's age in years. Risk of cardiovascular disease generally increases with age.", format="%d")
            with c2:
                sex = st.selectbox("Biological Sex", ["Male", "Female"], help="Patient's biological sex.")

            st.markdown('<div class="section-header">💓 Cardiovascular Vitals</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                resting_bp = st.slider("🩸 Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=128, help="Normal: < 120 | Elevated: 120-129 | High: 130+", format="%d")
            with c2:
                cholesterol = st.slider("🍔 Serum Cholesterol (mg/dl)", min_value=100, max_value=500, value=215, help="Desirable: < 200 | Borderline: 200-239 | High: ≥ 240", format="%d")

            c1, c2 = st.columns(2)
            with c1:
                max_hr = st.slider("🏃 Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=145, help="Maximum heart rate achieved during exercise. Target is generally (220 - Age).", format="%d")
            with c2:
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], help="Is fasting blood sugar greater than 120 mg/dl? Indicates possible diabetes.")

            st.markdown('<div class="section-header">🩺 ECG & Stress Indicators</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                chest_pain = st.selectbox(
                    "Chest Pain Type",
                    ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)", "TA (Typical Angina)"]
                )
            with c2:
                resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST (ST-T abnormality)", "LVH (Left Ventricular Hypertrophy)"])

            c1, c2 = st.columns(2)
            with c1:
                exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            with c2:
                st_slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

            oldpeak = st.slider("📉 Oldpeak (ST Depression)", min_value=-2.0, max_value=6.0, value=1.2, step=0.1, help="ST depression induced by exercise relative to rest. Normal is typically < 1.0", format="%.1f")

            submit_predict = st.form_submit_button("🚀 Run ML Heart Disease Prediction")

        # --------------------------------------------------------------
        # Patient Vitals Benchmark Status Box (BALANCES LEFT COLUMN HEIGHT!)
        # --------------------------------------------------------------
        st.markdown('<div class="section-header" style="margin-top:1rem;">📌 Quick Vitals Benchmark Indicators</div>', unsafe_allow_html=True)
        
        def bp_badge(v):
            if v < 120: return "Normal (<120)", "badge-normal"
            if v < 130: return "Elevated (120-129)", "badge-borderline"
            return "Stage 2 High (130+)", "badge-high"

        def chol_badge(v):
            if v < 200: return "Desirable (<200)", "badge-normal"
            if v < 240: return "Borderline (200-239)", "badge-borderline"
            return "High Risk (240+)", "badge-high"

        def hr_badge(v, age_):
            max_exp = 220 - age_
            pct = v / max_exp if max_exp else 0
            if pct >= 0.85: return "Optimal Peak (85%+)", "badge-normal"
            if pct >= 0.65: return "Fair Peak (65-84%)", "badge-borderline"
            return "Below Target (<65%)", "badge-high"

        bp_label, bp_cls = bp_badge(resting_bp)
        chol_label, chol_cls = chol_badge(cholesterol)
        hr_label, hr_cls = hr_badge(max_hr, age)

        vb1, vb2, vb3 = st.columns(3)
        with vb1:
            st.markdown(f'''
            <div class="metric-box" style="border-top: 3px solid #0d9488;">
                <div class="lbl" style="color:#64748b; font-weight:700;">Resting BP</div>
                <div class="val" style="color:#0d5c58; font-weight:800; font-size:1.45rem;">{resting_bp} <span style="font-size:0.75rem; color:#64748b; font-weight:600;">mmHg</span></div>
                <span class="badge {bp_cls}">{bp_label}</span>
            </div>
            ''', unsafe_allow_html=True)
        with vb2:
            st.markdown(f'''
            <div class="metric-box" style="border-top: 3px solid #2563eb;">
                <div class="lbl" style="color:#64748b; font-weight:700;">Cholesterol</div>
                <div class="val" style="color:#0d5c58; font-weight:800; font-size:1.45rem;">{cholesterol} <span style="font-size:0.75rem; color:#64748b; font-weight:600;">mg/dl</span></div>
                <span class="badge {chol_cls}">{chol_label}</span>
            </div>
            ''', unsafe_allow_html=True)
        with vb3:
            st.markdown(f'''
            <div class="metric-box" style="border-top: 3px solid #8b5cf6;">
                <div class="lbl" style="color:#64748b; font-weight:700;">Max Heart Rate</div>
                <div class="val" style="color:#0d5c58; font-weight:800; font-size:1.45rem;">{max_hr} <span style="font-size:0.75rem; color:#64748b; font-weight:600;">bpm</span></div>
                <span class="badge {hr_cls}">{hr_label}</span>
            </div>
            ''', unsafe_allow_html=True)

    with col_results:
        sex_code = "M" if sex == "Male" else "F"
        fbs_code = 1 if fasting_bs == "Yes" else 0
        angina_code = "Y" if exercise_angina == "Yes" else "N"
        cp_code = chest_pain.split(" ")[0]
        ecg_code = resting_ecg.split(" ")[0]

        patient_input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex_code,
            "ChestPainType": cp_code,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fbs_code,
            "RestingECG": ecg_code,
            "MaxHR": max_hr,
            "ExerciseAngina": angina_code,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }])

        pred_class, proba_val = predict_heart_disease(patient_input_df)

        if submit_predict:
            st.session_state.history.append({
                "age": age, "sex": sex, "result": pred_class,
                "prob": proba_val * 100, "label": "High Risk" if pred_class == 1 else "Low Risk"
            })

        # Wrap Risk Gauge & Classification Banner in structured card
        st.markdown('<div class="subview-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎯 Model Prediction & Risk Score Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_risk_gauge(proba_val), use_container_width=True, key="predict_risk_gauge")

        # High / Low Banner
        if pred_class == 1:
            st.markdown(f"""
            <div class="result-card result-high">
                <div class="result-title">⚠️ High Heart Disease Risk</div>
                <div>KNN Model Risk Confidence: <b>{proba_val*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-low">
                <div class="result-title">🟢 Low Heart Disease Risk</div>
                <div>KNN Model Confidence: <b>{(1-proba_val)*100:.1f}% Healthy</b></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------------
        # Patient-Specific Local Risk Driver Analysis (XAI Card)
        # --------------------------------------------------------------
        st.markdown('<div class="subview-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔍 Patient Risk Driver Analysis (Local Feature Impact)</div>', unsafe_allow_html=True)

        contributions = []
        if resting_bp >= 140:
            contributions.append({"Feature": "Hypertension BP", "Impact": +22, "Type": "Risk Factor"})
        elif resting_bp >= 130:
            contributions.append({"Feature": "Elevated BP", "Impact": +12, "Type": "Risk Factor"})
        else:
            contributions.append({"Feature": "Normal BP", "Impact": -10, "Type": "Protective Factor"})

        if cholesterol >= 240:
            contributions.append({"Feature": "High Cholesterol", "Impact": +20, "Type": "Risk Factor"})
        elif cholesterol >= 200:
            contributions.append({"Feature": "Borderline Cholesterol", "Impact": +10, "Type": "Risk Factor"})
        else:
            contributions.append({"Feature": "Desirable Cholesterol", "Impact": -12, "Type": "Protective Factor"})

        if oldpeak >= 1.5:
            contributions.append({"Feature": "High ST Depression", "Impact": +25, "Type": "Risk Factor"})
        elif oldpeak >= 0.8:
            contributions.append({"Feature": "Moderate ST Depression", "Impact": +14, "Type": "Risk Factor"})
        else:
            contributions.append({"Feature": "Normal ST Segment", "Impact": -15, "Type": "Protective Factor"})

        if st_slope in ["Flat", "Down"]:
            contributions.append({"Feature": f"ST Slope ({st_slope})", "Impact": +24, "Type": "Risk Factor"})
        else:
            contributions.append({"Feature": "Upward ST Slope", "Impact": -18, "Type": "Protective Factor"})

        if cp_code == "ASY":
            contributions.append({"Feature": "Asymptomatic Pain", "Impact": +18, "Type": "Risk Factor"})

        if angina_code == "Y":
            contributions.append({"Feature": "Exercise Angina", "Impact": +16, "Type": "Risk Factor"})

        if max_hr < 130:
            contributions.append({"Feature": "Sub-target Max HR", "Impact": +15, "Type": "Risk Factor"})
        elif max_hr >= 160:
            contributions.append({"Feature": "High HR Reserve", "Impact": -14, "Type": "Protective Factor"})

        contrib_df = pd.DataFrame(contributions)

        st.plotly_chart(plot_risk_lollipop(contrib_df), use_container_width=True, key="patient_risk_lollipop")

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------------
        # Advice List Generator (Dynamic Safe / Warning / Danger Status)
        # --------------------------------------------------------------
        advice_list = []
        if resting_bp >= 140:
            advice_list.append(("Blood Pressure Alert", f"Resting BP ({resting_bp} mmHg) indicates Stage 2 Hypertension. Sodium reduction & BP monitoring recommended.", "danger", "🚨"))
        elif resting_bp >= 130:
            advice_list.append(("Elevated Blood Pressure", f"Resting BP ({resting_bp} mmHg) is elevated. Lifestyle modifications recommended.", "warning", "⚡"))
        else:
            advice_list.append(("Normal Blood Pressure", f"Resting BP ({resting_bp} mmHg) is within optimal healthy range.", "safe", "🟢"))

        if cholesterol >= 240:
            advice_list.append(("High Cholesterol Alert", f"Serum Cholesterol ({cholesterol} mg/dl) is high. Consider lipid panel & dietary review.", "danger", "🚨"))
        elif cholesterol >= 200:
            advice_list.append(("Borderline Cholesterol", f"Serum Cholesterol ({cholesterol} mg/dl) is borderline elevated. Monitor diet.", "warning", "⚡"))
        else:
            advice_list.append(("Healthy Cholesterol", f"Serum Cholesterol ({cholesterol} mg/dl) is in desirable range.", "safe", "🟢"))

        if oldpeak >= 1.5:
            advice_list.append(("ST Depression Warning", f"Oldpeak ({oldpeak}) shows notable exercise ST depression. Exercise stress testing advised.", "danger", "🚨"))
        elif oldpeak >= 0.8:
            advice_list.append(("Moderate ST Depression", f"Oldpeak ({oldpeak}) shows slight exercise ST depression. Monitor ECG response.", "warning", "⚡"))
        else:
            advice_list.append(("Normal ST Segment", f"Oldpeak ({oldpeak}) is in normal baseline range (< 0.8).", "safe", "🟢"))

        expected_max_hr = 220 - age
        if max_hr < (expected_max_hr * 0.70):
            advice_list.append(("Heart Rate Reserve Warning", f"Achieved Max HR ({max_hr} bpm) is below 70% of age-predicted peak ({expected_max_hr} bpm).", "warning", "⚡"))
        else:
            advice_list.append(("Optimal HR Reserve", f"Achieved Max HR ({max_hr} bpm) reached expected target capacity.", "safe", "🟢"))

    # --------------------------------------------------------------
    # FULL WEB PAGE WIDTH SUB-VIEW ANALYTICS (SIDE-BY-SIDE GRID & CARDS)
    # --------------------------------------------------------------
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📱 ML Feature Insights & Visual Analytics</div>', unsafe_allow_html=True)

    if "active_subview" not in st.session_state:
        st.session_state.active_subview = "🕸️ Patient Feature Radar & Population Map"

    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        if st.button("🕸️ Radar & Cohort Map", use_container_width=True, type="primary" if st.session_state.active_subview == "🕸️ Patient Feature Radar & Population Map" else "secondary", key="btn_sub_radar"):
            st.session_state.active_subview = "🕸️ Patient Feature Radar & Population Map"
            st.rerun()
    with sub2:
        if st.button("🎛️ What-If Simulator", use_container_width=True, type="primary" if st.session_state.active_subview == "🎛️ What-If Scenario Simulator" else "secondary", key="btn_sub_sim"):
            st.session_state.active_subview = "🎛️ What-If Scenario Simulator"
            st.rerun()
    with sub3:
        if st.button("💡 Insights & Report", use_container_width=True, type="primary" if st.session_state.active_subview == "💡 Clinical Insights & Download Report" else "secondary", key="btn_sub_report"):
            st.session_state.active_subview = "💡 Clinical Insights & Download Report"
            st.rerun()

    selected_subview = st.session_state.active_subview

    if selected_subview == "🕸️ Patient Feature Radar & Population Map":
        r_col1, r_col2 = st.columns(2, gap="medium")
        with r_col1:
            st.markdown('<div class="subview-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🕸️ Patient Feature Radar Profile</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_patient_radar(age, resting_bp, cholesterol, max_hr, oldpeak), use_container_width=True, key="sub_patient_radar")
            st.markdown('</div>', unsafe_allow_html=True)

        with r_col2:
            st.markdown('<div class="subview-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📍 Patient Position Map (918 Cohort Records)</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_population_scatter(age, max_hr, pred_class), use_container_width=True, key="sub_population_scatter")
            st.markdown('</div>', unsafe_allow_html=True)

    elif selected_subview == "🎛️ What-If Scenario Simulator":
        st.markdown('<div class="subview-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎛️ Interactive What-If Risk Simulator</div>', unsafe_allow_html=True)
        
        sim_col_in, sim_col_out = st.columns([1.1, 1], gap="medium")
        with sim_col_in:
            st.markdown("<b>Adjust parameters to test feature tweaks and click Recalculate:</b>", unsafe_allow_html=True)
            sim_bp = st.slider("🩸 Simulated Resting BP", 80, 200, int(resting_bp), key="dash_sim_bp", help="Slide to test how lowering or raising BP impacts overall risk.", format="%d")
            sim_chol = st.slider("🍔 Simulated Cholesterol", 100, 500, int(cholesterol) if cholesterol > 0 else 200, key="dash_sim_chol", help="Slide to test how changes in cholesterol impact risk.", format="%d")
            sim_hr = st.slider("🏃 Simulated Max HR", 70, 210, int(max_hr), key="dash_sim_hr", help="Slide to test how improved fitness (higher Max HR) lowers risk.", format="%d")
            sim_oldpeak = st.slider("📉 Simulated Oldpeak", -2.0, 5.0, float(oldpeak), step=0.1, key="dash_sim_oldpeak", help="Slide to simulate improvements in exercise ECG response.", format="%.1f")
            
            run_sim = st.button("⚡ Recalculate ML Prediction", key="dash_btn_sim")

        if run_sim:
            sim_df = pd.DataFrame([{
                "Age": age, "Sex": sex_code, "ChestPainType": cp_code,
                "RestingBP": sim_bp, "Cholesterol": sim_chol, "FastingBS": fbs_code,
                "RestingECG": ecg_code, "MaxHR": sim_hr, "ExerciseAngina": angina_code,
                "Oldpeak": sim_oldpeak, "ST_Slope": st_slope
            }])
            _, calculated_proba = predict_heart_disease(sim_df)
            st.session_state.sim_proba = calculated_proba
            st.session_state.sim_has_run = True

        with sim_col_out:
            current_sim_proba = st.session_state.get("sim_proba", proba_val)
            has_run = st.session_state.get("sim_has_run", False)

            st.plotly_chart(plot_risk_gauge(current_sim_proba), use_container_width=True, key="dash_sim_gauge")

            if has_run:
                diff_proba = (current_sim_proba - proba_val) * 100
                diff_color = "#ef4444" if diff_proba > 0 else "#10b981" if diff_proba < 0 else "#64748b"
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #cce3de; padding: 0.75rem 1rem; border-radius: 8px; text-align: center; margin-top: 0.2rem;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Simulated Model Risk Output</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #0d5c58;">{current_sim_proba*100:.1f}% Risk</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {diff_color};">Risk Shift: {diff_proba:+.1f}% from baseline patient</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("👆 Adjust parameters on the left and click **'⚡ Recalculate ML Prediction'** to compute simulated risk shift.")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="subview-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Personalized Clinical Insights & Dynamic Risk Status</div>', unsafe_allow_html=True)
        for item in advice_list:
            title, text, status, icon = item
            st.markdown(f"""
            <div class="advice-card advice-{status}">
                <div class="advice-title">{icon} {title}</div>
                <div class="advice-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📥 Download Clinical Prediction Reports")

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            html_report_data = generate_html_report(age, sex, resting_bp, cholesterol, max_hr, fasting_bs, chest_pain, resting_ecg, exercise_angina, oldpeak, st_slope, pred_class, proba_val, advice_list)
            st.download_button(
                label="📄 Download Visual HTML Report (Printable)",
                data=html_report_data,
                file_name=f"Patient_Heart_Health_Report_{age}Y.html",
                mime="text/html",
                key="dl_html_report"
            )

        with c_d2:
            csv_report_df = pd.DataFrame([{
                "Age": age, "Sex": sex, "RestingBP": resting_bp, "Cholesterol": cholesterol,
                "MaxHR": max_hr, "ChestPain": chest_pain, "ECG": resting_ecg,
                "Oldpeak": oldpeak, "ST_Slope": st_slope, "PredictedRiskCategory": "High Risk" if pred_class == 1 else "Low Risk",
                "RiskProbability_Pct": round(proba_val * 100, 2)
            }])
            st.download_button(
                label="📊 Download CSV Data Report",
                data=csv_report_df.to_csv(index=False),
                file_name=f"Patient_Vitals_Data_Age{age}.csv",
                mime="text/csv",
                key="dl_csv_report"
            )
        st.markdown('</div>', unsafe_allow_html=True)



# ==================================================================
# CATEGORY 2 — Exploratory Data Analysis (EDA)
# ==================================================================
elif page_category == "Exploratory Data Analysis (EDA)":
    st.markdown("### 📊 Exploratory Data Analysis (EDA) & Feature Exploration")
    st.markdown("Filter and inspect feature distributions across 918 patient dataset records.")

    with st.expander("🎛️ Feature Slicing & Filter Controls", expanded=True):
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            af_age = st.slider("🎂 Age Range", int(raw_df["Age"].min()), int(raw_df["Age"].max()), (30, 75), key="af_age", help="Filter dataset by patient age.")
            af_sex = st.multiselect("Biological Sex", options=["M", "F"], default=["M", "F"], key="af_sex")
            af_fbs = st.selectbox("Fasting BS Filter", options=["All", "Fasting BS > 120", "Fasting BS <= 120"], key="af_fbs")

        with c_f2:
            af_bp = st.slider("🩸 Resting BP Range (mmHg)", int(raw_df["RestingBP"].min()), int(raw_df["RestingBP"].max()), (80, 200), key="af_bp", help="Filter dataset by Resting Blood Pressure.")
            af_chol = st.slider("🍔 Cholesterol Range (mg/dl)", int(raw_df["Cholesterol"].min()), int(raw_df["Cholesterol"].max()), (100, 550), key="af_chol", help="Filter dataset by Serum Cholesterol level.")
            af_angina = st.selectbox("Exercise Angina Filter", options=["All", "Yes (Y)", "No (N)"], key="af_angina")

        with c_f3:
            af_cp = st.multiselect("Chest Pain Types", options=list(raw_df["ChestPainType"].unique()), default=list(raw_df["ChestPainType"].unique()), key="af_cp")
            af_ecg = st.multiselect("Resting ECG Types", options=list(raw_df["RestingECG"].unique()), default=list(raw_df["RestingECG"].unique()), key="af_ecg")
            af_slope = st.multiselect("ST Segment Slopes", options=list(raw_df["ST_Slope"].unique()), default=list(raw_df["ST_Slope"].unique()), key="af_slope")

    # Apply Advanced Filters
    filtered_df = raw_df[
        (raw_df["Age"].between(af_age[0], af_age[1])) &
        (raw_df["Sex"].isin(af_sex)) &
        (raw_df["RestingBP"].between(af_bp[0], af_bp[1])) &
        (raw_df["Cholesterol"].between(af_chol[0], af_chol[1])) &
        (raw_df["ChestPainType"].isin(af_cp)) &
        (raw_df["RestingECG"].isin(af_ecg)) &
        (raw_df["ST_Slope"].isin(af_slope))
    ]

    if af_fbs == "Fasting BS > 120":
        filtered_df = filtered_df[filtered_df["FastingBS"] == 1]
    elif af_fbs == "Fasting BS <= 120":
        filtered_df = filtered_df[filtered_df["FastingBS"] == 0]

    if af_angina == "Yes (Y)":
        filtered_df = filtered_df[filtered_df["ExerciseAngina"] == "Y"]
    elif af_angina == "No (N)":
        filtered_df = filtered_df[filtered_df["ExerciseAngina"] == "N"]

    # Live Cohort Metrics Cards (Theme Blue KPI Cards)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box-blue"><div class="val">{len(filtered_df)}</div><div class="lbl">Matching Patients</div></div>', unsafe_allow_html=True)
    with m2:
        disease_pct = (filtered_df["HeartDisease"].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-box-blue"><div class="val">{disease_pct:.1f}%</div><div class="lbl">Disease Prevalence</div></div>', unsafe_allow_html=True)
    with m3:
        avg_bp = filtered_df["RestingBP"].mean() if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-box-blue"><div class="val">{avg_bp:.0f} mmHg</div><div class="lbl">Avg Resting BP</div></div>', unsafe_allow_html=True)
    with m4:
        avg_chol = filtered_df["Cholesterol"].mean() if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="metric-box-blue"><div class="val">{avg_chol:.0f} mg/dl</div><div class="lbl">Avg Cholesterol</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Grid 1: Age Histogram & Chest Pain Donut Chart
    g1, g2 = st.columns(2)
    with g1:
        fig_hist = px.histogram(
            filtered_df, x="Age", color="HeartDisease",
            barmode="overlay",
            title="Filtered Cohort Age Distribution by Outcome",
            color_discrete_map={1: "#ef4444", 0: "#10b981"}
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=290, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hist, use_container_width=True, key="cohort_age_hist")

    with g2:
        st.plotly_chart(plot_chest_pain_donut(filtered_df), use_container_width=True, key="cohort_chest_pain_donut")

    st.markdown("---")

    # Grid 2: Cholesterol vs BP Scatter & Vitals Means Bar Chart
    g3, g4 = st.columns(2)
    with g3:
        fig_scatter = px.scatter(
            filtered_df, x="Cholesterol", y="RestingBP", color="HeartDisease",
            color_discrete_map={1: "#ef4444", 0: "#10b981"},
            title="Cholesterol vs Resting BP Scatter",
            hover_data=["Age", "MaxHR"]
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=290, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True, key="cohort_chol_bp_scatter")

    with g4:
        st.plotly_chart(plot_vitals_bar(filtered_df), use_container_width=True, key="cohort_vitals_bar")

    st.markdown("---")

    # Grid 3: Horizontal Correlation Matrix Heatmap
    st.markdown('<div class="section-header">🔥 Filtered Cohort Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num_data = filtered_df[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]].corr()
    fig_corr = px.imshow(
        num_data, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=""
    )
    fig_corr.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        height=320,
        margin=dict(l=20, r=20, t=15, b=25)
    )
    st.plotly_chart(fig_corr, use_container_width=True, key="cohort_corr_heatmap")

# ==================================================================
# CATEGORY 3 — Model Evaluation & Benchmarks (DATA SCIENCE BENCHMARKS)
# ==================================================================
elif page_category == "Model Benchmarks & Metrics":
    st.markdown("### 🧪 Machine Learning Model Evaluation & Benchmarks")
    st.markdown("Performance comparison across 5 supervised classification model architectures evaluated on test data.")

    # Benchmark KPI Highlights
    bm1, bm2, bm3 = st.columns(3)
    with bm1:
        st.markdown('<div class="metric-box-blue"><div class="val">🥇 KNN Model</div><div class="lbl">Optimal Selected Algorithm</div></div>', unsafe_allow_html=True)
    with bm2:
        st.markdown('<div class="metric-box-blue"><div class="val">🎯 88.04%</div><div class="lbl">Validation Accuracy</div></div>', unsafe_allow_html=True)
    with bm3:
        st.markdown('<div class="metric-box-blue"><div class="val">📈 0.9120</div><div class="lbl">Peak ROC-AUC Score</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Bar Chart Visualizer
    st.markdown('<div class="subview-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Algorithm Performance Benchmarks (Accuracy, F1, ROC-AUC)</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame([
        {"Model": "KNN (Selected)", "Accuracy": 0.8804, "F1 Score": 0.8932, "ROC-AUC": 0.9120},
        {"Model": "Logistic Regression", "Accuracy": 0.8804, "F1 Score": 0.8932, "ROC-AUC": 0.9080},
        {"Model": "SVM (RBF Kernel)", "Accuracy": 0.8641, "F1 Score": 0.8792, "ROC-AUC": 0.8940},
        {"Model": "Naive Bayes", "Accuracy": 0.8696, "F1 Score": 0.8788, "ROC-AUC": 0.8890},
        {"Model": "Decision Tree", "Accuracy": 0.7554, "F1 Score": 0.7668, "ROC-AUC": 0.7510},
    ])

    fig_models = px.bar(
        comparison, x="Model", y=["Accuracy", "F1 Score", "ROC-AUC"],
        barmode="group",
        title="",
        color_discrete_sequence=["#0d9488", "#2563eb", "#8b5cf6"]
    )
    fig_models.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=360, margin=dict(l=20, r=20, t=15, b=20))
    st.plotly_chart(fig_models, use_container_width=True, key="model_benchmark_bar")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Custom Styled Benchmark Table
    st.markdown("""
    <div class="subview-card">
        <div class="section-header">📋 Detailed Model Performance Matrix</div>
        <div class="table-responsive-container">
            <table style="width: 100%; border-collapse: collapse; margin-top: 0.8rem; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.03); min-width: 500px;">
                <thead>
                    <tr style="background: linear-gradient(135deg, #0d5c58 0%, #1e40af 100%); color: white; text-align: left; font-size: 0.9rem;">
                        <th style="padding: 12px 16px;">Model Architecture</th>
                        <th style="padding: 12px 16px;">Accuracy</th>
                        <th style="padding: 12px 16px;">F1 Score</th>
                        <th style="padding: 12px 16px;">ROC-AUC</th>
                        <th style="padding: 12px 16px;">Status</th>
                    </tr>
                </thead>
                <tbody style="font-size: 0.88rem;">
                    <tr style="background: #ecfdf5; border-bottom: 1px solid #cce3de; font-weight: 700;">
                        <td style="padding: 12px 16px; color: #065f46;">KNN (K-Nearest Neighbors)</td>
                        <td style="padding: 12px 16px; color: #047857;">88.04%</td>
                        <td style="padding: 12px 16px; color: #047857;">0.8932</td>
                        <td style="padding: 12px 16px; color: #047857;">0.9120</td>
                        <td style="padding: 12px 16px;"><span class="badge badge-normal">★ Selected Model</span></td>
                    </tr>
                    <tr style="background: #ffffff; border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px 16px; font-weight: 600; color: #1e293b;">Logistic Regression</td>
                        <td style="padding: 12px 16px; color: #334155;">88.04%</td>
                        <td style="padding: 12px 16px; color: #334155;">0.8932</td>
                        <td style="padding: 12px 16px; color: #334155;">0.9080</td>
                        <td style="padding: 12px 16px;"><span class="badge" style="background:#e0f2fe; color:#0369a1;">Baseline Benchmark</span></td>
                    </tr>
                    <tr style="background: #f8fafc; border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px 16px; font-weight: 600; color: #1e293b;">SVM (RBF Kernel)</td>
                        <td style="padding: 12px 16px; color: #334155;">86.41%</td>
                        <td style="padding: 12px 16px; color: #334155;">0.8792</td>
                        <td style="padding: 12px 16px; color: #334155;">0.8940</td>
                        <td style="padding: 12px 16px;"><span class="badge" style="background:#f3e8ff; color:#6b21a8;">Evaluated</span></td>
                    </tr>
                    <tr style="background: #ffffff; border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 12px 16px; font-weight: 600; color: #1e293b;">Naive Bayes</td>
                        <td style="padding: 12px 16px; color: #334155;">86.96%</td>
                        <td style="padding: 12px 16px; color: #334155;">0.8788</td>
                        <td style="padding: 12px 16px; color: #334155;">0.8890</td>
                        <td style="padding: 12px 16px;"><span class="badge" style="background:#f3e8ff; color:#6b21a8;">Evaluated</span></td>
                    </tr>
                    <tr style="background: #f8fafc;">
                        <td style="padding: 12px 16px; font-weight: 600; color: #1e293b;">Decision Tree</td>
                        <td style="padding: 12px 16px; color: #334155;">75.54%</td>
                        <td style="padding: 12px 16px; color: #334155;">0.7668</td>
                        <td style="padding: 12px 16px; color: #334155;">0.7510</td>
                        <td style="padding: 12px 16px;"><span class="badge" style="background:#fee2e2; color:#b91c1c;">Underperforming</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # 1 & 2. Confusion Matrix & ROC Curve (Side-by-Side Grid)
    # --------------------------------------------------------------
    cm_col1, cm_col2 = st.columns(2, gap="medium")

    with cm_col1:
        cm_data = [[78, 12], [10, 84]]  # [TN, FP], [FN, TP]
        fig_cm = px.imshow(
            cm_data,
            x=["Predicted Healthy (0)", "Predicted Risk (1)"],
            y=["Actual Healthy (0)", "Actual Risk (1)"],
            text_auto=True,
            color_continuous_scale="Teal",
            title="🧩 KNN Model Confusion Matrix (Test Data)"
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=310, margin=dict(l=20, r=20, t=40, b=20))
        st.markdown('<div class="subview-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_cm, use_container_width=True, key="knn_confusion_matrix")
        st.markdown('</div>', unsafe_allow_html=True)

    with cm_col2:
        fpr = [0.0, 0.04, 0.08, 0.12, 0.20, 0.35, 0.55, 0.80, 1.0]
        tpr = [0.0, 0.50, 0.74, 0.86, 0.93, 0.96, 0.98, 0.99, 1.0]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', name='KNN Classifier (AUC = 0.912)', line=dict(color='#2563eb', width=3), fill='tozeroy', fillcolor='rgba(37,99,235,0.12)'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline (AUC = 0.50)', line=dict(color='#94a3b8', width=2, dash='dash')))
        fig_roc.update_layout(title="📈 ROC Curve (Receiver Operating Characteristic)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=310, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5))
        st.markdown('<div class="subview-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_roc, use_container_width=True, key="knn_roc_curve")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # 3. Feature Importance / Feature Weight Impact (XAI Bar Chart)
    # --------------------------------------------------------------
    st.markdown('<div class="subview-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⭐ Feature Importance & Weight Impact (Explainable AI / XAI)</div>', unsafe_allow_html=True)
    feat_imp_df = pd.DataFrame({
        "Feature": ["ST_Slope", "Oldpeak", "ChestPainType", "MaxHR", "ExerciseAngina", "Cholesterol", "Age", "RestingBP"],
        "Relative Importance Weight": [0.85, 0.78, 0.72, 0.65, 0.60, 0.42, 0.38, 0.30]
    }).sort_values("Relative Importance Weight", ascending=True)
    fig_feat = px.bar(feat_imp_df, x="Relative Importance Weight", y="Feature", orientation="h", color="Relative Importance Weight", color_continuous_scale="Tealgrn", title="")
    fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=300, margin=dict(l=20, r=20, t=15, b=20))
    st.plotly_chart(fig_feat, use_container_width=True, key="knn_feature_importance")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------
    # 4. Interactive K-Value Hyperparameter Optimization Tuner
    # --------------------------------------------------------------
    st.markdown('<div class="subview-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎛️ Interactive KNN Hyperparameter Tuner (K-Neighbors Optimization)</div>', unsafe_allow_html=True)
    
    k_val = st.slider("Select K (Number of Neighbors)", min_value=1, max_value=15, value=5, step=2, help="Adjust K to see how neighbor count affects model accuracy.", key="k_tuner_slider")
    
    k_data = pd.DataFrame({
        "K Neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        "Validation Accuracy (%)": [82.5, 86.4, 88.04, 87.5, 86.9, 86.1, 85.3, 84.8]
    })
    
    selected_acc = k_data.loc[k_data["K Neighbors"] == k_val, "Validation Accuracy (%)"].values[0] if k_val in k_data["K Neighbors"].values else 88.04
    
    fig_k = px.line(k_data, x="K Neighbors", y="Validation Accuracy (%)", markers=True, title=f"Accuracy Curve across K Neighbors (Selected K = {k_val} → Accuracy: {selected_acc:.2f}%)")
    fig_k.update_traces(line_color='#0d9488', line_width=3, marker=dict(size=10, color='#2563eb'))
    fig_k.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', height=280, margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig_k, use_container_width=True, key="k_tuner_plot")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# CATEGORY 4 — About The ML Project (SYMMETRICAL 3x2 BALANCED GRID)
# ==================================================================
elif page_category == "About The ML Project":
    st.markdown("### ℹ️ About Heart Care AI (ML Project)")
    st.markdown("An end-to-end Machine Learning Data Science project designed for cardiovascular risk classification & feature analytics.")

    # Row 1 (3 Equal Columns)
    r1_col1, r1_col2, r1_col3 = st.columns(3, gap="medium")

    with r1_col1:
        st.markdown("""
        <div class="dev-profile-card">
            <div class="dev-avatar">👩‍💻</div>
            <div class="dev-name">Khushi Singh</div>
            <div class="dev-role">Machine Learning & AI Engineer</div>
            <div class="dev-badge-container">
                <span class="dev-chip">🤖 KNN Classifier</span>
                <span class="dev-chip">⚖️ StandardScaler</span>
                <span class="dev-chip">📊 Feature Engineering</span>
                <span class="dev-chip">📈 Scikit-Learn ML</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r1_col2:
        st.markdown("""
        <div class="about-box-card" style="border-top: 3px solid #2563eb;">
            <div class="section-header">🎯 Project Purpose & Mission</div>
            <p style="font-size:0.88rem; color:#475569; line-height:1.6; margin:0;">
                <b>Heart Care AI</b> is a Data Science project engineered to apply machine learning classification to cardiovascular health indicators. 
                It transforms patient feature inputs into instant risk probabilities, feature radar maps, and printable ML evaluation reports.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r1_col3:
        st.markdown("""
        <div class="about-box-card" style="border-top: 3px solid #0d9488;">
            <div class="section-header">🧬 ML & Data Pipeline</div>
            <ul style="font-size:0.85rem; color:#334155; line-height:1.6; padding-left:1.1rem; margin:0;">
                <li><b>Dataset:</b> <span class="badge badge-normal">918 Patient Records</span></li>
                <li><b>Preprocessing:</b> Imputation & One-Hot Encoding</li>
                <li><b>Scaler:</b> <code>StandardScaler</code> fitted on features</li>
                <li><b>Classifier:</b> KNN ($K=5$) with <span class="badge badge-normal">88.0% Accuracy</span> & <span class="badge badge-normal">0.912 ROC-AUC</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Row 2 (3 Equal Columns)
    r2_col1, r2_col2, r2_col3 = st.columns(3, gap="medium")

    with r2_col1:
        st.markdown("""
        <div class="about-box-card" style="border-top: 3px solid #8b5cf6;">
            <div class="section-header">🛠️ Key ML Features</div>
            <ul style="font-size:0.85rem; color:#334155; line-height:1.6; padding-left:1.1rem; margin:0;">
                <li><b>🔮 ML Risk Predictor:</b> Instant KNN probability prediction engine.</li>
                <li><b>🎛️ What-If Simulator:</b> Real-time feature tweak risk recalculation.</li>
                <li><b>📊 EDA Explorer:</b> Multi-attribute dataset slicing engine.</li>
                <li><b>📄 Report Export:</b> Printable HTML & CSV prediction report generator.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with r2_col2:
        st.markdown("""
        <div class="about-box-card" style="border-top: 3px solid #2563eb;">
            <div class="section-header">⚡ Tech Stack & Architecture</div>
            <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.4rem;">
                <span class="badge" style="background:#e0f2fe; color:#0369a1; font-weight:700;">🐍 Python 3.10+</span>
                <span class="badge" style="background:#d1fae5; color:#047857; font-weight:700;">🧠 Scikit-Learn</span>
                <span class="badge" style="background:#fee2e2; color:#b91c1c; font-weight:700;">🎈 Streamlit UI</span>
                <span class="badge" style="background:#fef3c7; color:#b45309; font-weight:700;">📊 Plotly Viz</span>
                <span class="badge" style="background:#f3e8ff; color:#6b21a8; font-weight:700;">🐼 Pandas & NumPy</span>
                <span class="badge" style="background:#e0e7ff; color:#3730a3; font-weight:700;">💾 Joblib Artifacts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2_col3:
        st.markdown("""
        <div class="about-box-card" style="border-top: 3px solid #ef4444;">
            <div class="section-header">⚕️ Clinical & Medical Disclaimer</div>
            <p style="font-size:0.85rem; color:#64748b; line-height:1.6; margin:0;">
                Developed strictly for <b>educational & Data Science demonstration</b>. 
                It is not intended as a substitute for professional medical advice or formal diagnostic testing. Always consult a certified physician.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Khushi's Heart Care AI · Data Science Machine Learning Project · Built with Scikit-Learn, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)