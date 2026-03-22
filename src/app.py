"""
AI-Powered Customer Churn Prediction Dashboard
================================================
Interactive Streamlit application for predicting customer churn
and generating AI-powered retention recommendations.

Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Attempt to import AI module
try:
    from ai_insights import generate_ai_insights
except ImportError:
    from src.ai_insights import generate_ai_insights

# ── Page config ──
st.set_page_config(page_title="Churn Predictor AI", page_icon="📊", layout="wide")

# ── Load artifacts ──
@st.cache_resource
def load_model():
    base = Path(__file__).parent / "artifacts"
    if not base.exists():
        base = Path("src/artifacts")
    with open(base / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(base / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(base / "features.json") as f:
        features = json.load(f)
    with open(base / "metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, features, metrics

@st.cache_data
def load_data():
    for p in ["data/customers.csv", "../data/customers.csv"]:
        if Path(p).exists():
            return pd.read_csv(p)
    return None

model, scaler, feature_cols, metrics = load_model()
df = load_data()

# ── Sidebar ──
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Individual Prediction", "AI Retention Insights", "Model Performance"])

# ══════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════
if page == "Dashboard Overview":
    st.title("📊 Customer Churn Analytics Dashboard")
    st.markdown("Real-time churn risk analysis across the Enterprise Mobility UK customer base.")

    if df is not None:
        churned = df["churned"].sum()
        total = len(df)
        rate = churned / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("Churned", f"{churned:,}", delta=f"{rate:.1%}", delta_color="inverse")
        c3.metric("Retained", f"{total - churned:,}")
        c4.metric("Avg Tenure", f"{df['tenure_months'].mean():.0f} months")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Rate by Region")
            region_churn = df.groupby("region")["churned"].mean().sort_values(ascending=True).reset_index()
            region_churn.columns = ["Region", "Churn Rate"]
            fig = px.bar(region_churn, x="Churn Rate", y="Region", orientation="h",
                         color="Churn Rate", color_continuous_scale="RdYlGn_r")
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Churn by Age Group")
            df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 50, 66],
                                     labels=["18-25", "26-35", "36-50", "51-65"])
            age_churn = df.groupby("age_group")["churned"].mean().reset_index()
            age_churn.columns = ["Age Group", "Churn Rate"]
            fig2 = px.bar(age_churn, x="Age Group", y="Churn Rate",
                          color="Churn Rate", color_continuous_scale="RdYlGn_r")
            fig2.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Digital Engagement vs Churn")
            fig3 = px.histogram(df, x="digital_engagement_score", color="churned",
                                barmode="overlay", nbins=30,
                                labels={"churned": "Churned", "digital_engagement_score": "Digital Engagement"},
                                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
            fig3.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.subheader("Satisfaction vs Churn")
            sat_df = df.groupby(["pricing_satisfaction", "churned"]).size().reset_index(name="count")
            fig4 = px.bar(sat_df, x="pricing_satisfaction", y="count", color="churned",
                          barmode="group", labels={"pricing_satisfaction": "Pricing Satisfaction (1-10)"},
                          color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
            fig4.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 2: INDIVIDUAL PREDICTION
# ══════════════════════════════════════════════════════
elif page == "Individual Prediction":
    st.title("🎯 Individual Churn Prediction")
    st.markdown("Enter customer details to predict their churn risk.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            tenure = st.number_input("Tenure (months)", 1, 120, 12)
            spend = st.number_input("Monthly Spend (£)", 10.0, 1000.0, 80.0)
        with col2:
            rentals = st.number_input("Rentals (12 months)", 0, 50, 6)
            digital = st.slider("Digital Engagement", 0.0, 1.0, 0.3, 0.01)
            price_sat = st.slider("Pricing Satisfaction", 1, 10, 6)
            svc_sat = st.slider("Service Satisfaction", 1, 10, 7)
        with col3:
            nps = st.slider("NPS Score", -100, 100, 20)
            loyal = st.selectbox("Loyalty Member", ["Yes", "No"])
            complaints = st.number_input("Complaints (6 months)", 0, 10, 0)
            region = st.selectbox("Region", ["London", "South East", "South West", "Midlands",
                                              "North West", "North East", "Scotland", "Wales", "East Anglia"])

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        from sklearn.preprocessing import LabelEncoder
        # Build feature vector matching training features
        profile = {
            "age": age, "gender": gender, "tenure_months": tenure,
            "monthly_spend_gbp": spend, "num_rentals_12m": rentals,
            "digital_engagement_score": digital, "pricing_satisfaction": price_sat,
            "service_satisfaction": svc_sat, "nps_score": nps,
            "loyalty_member": 1 if loyal == "Yes" else 0,
            "complaints_last_6m": complaints, "region": region,
        }

        # Engineered features
        profile["spend_per_rental"] = spend / (rentals + 1)
        profile["satisfaction_avg"] = (price_sat + svc_sat) / 2
        profile["is_young"] = 1 if age < 26 else 0
        profile["high_complaints"] = 1 if complaints >= 2 else 0
        profile["low_engagement"] = 1 if digital < 0.2 else 0

        # Encode
        gender_map = {"Female": 0, "Male": 1, "Other": 2}
        region_list = sorted(["London", "South East", "South West", "Midlands", "North West",
                              "North East", "Scotland", "Wales", "East Anglia"])
        region_map = {r: i for i, r in enumerate(region_list)}
        tenure_bins = ["0-6m", "1-2y", "2-5y", "5y+", "6-12m"]
        if tenure <= 6: tbin = "0-6m"
        elif tenure <= 12: tbin = "6-12m"
        elif tenure <= 24: tbin = "1-2y"
        elif tenure <= 60: tbin = "2-5y"
        else: tbin = "5y+"
        tbin_map = {b: i for i, b in enumerate(sorted(tenure_bins))}

        profile["gender_enc"] = gender_map.get(gender, 1)
        profile["region_enc"] = region_map.get(region, 0)
        profile["tenure_bin_enc"] = tbin_map.get(tbin, 0)

        X_input = np.array([[profile[f] for f in feature_cols]])
        prob = model.predict_proba(X_input)[0][1]

        st.markdown("---")
        risk = "🔴 HIGH" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.4 else "🟢 LOW"

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.metric("Churn Probability", f"{prob:.1%}")
            st.metric("Risk Level", risk)
        with rc2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Risk Score"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#e74c3c" if prob > 0.7 else "#f39c12" if prob > 0.4 else "#2ecc71"},
                       "steps": [
                           {"range": [0, 40], "color": "#d5f5e3"},
                           {"range": [40, 70], "color": "#fdebd0"},
                           {"range": [70, 100], "color": "#fadbd8"}]}))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Store in session for AI insights page
        st.session_state["last_profile"] = profile
        st.session_state["last_prob"] = prob


# ══════════════════════════════════════════════════════
# PAGE 3: AI RETENTION INSIGHTS
# ══════════════════════════════════════════════════════
elif page == "AI Retention Insights":
    st.title("🤖 AI-Powered Retention Recommendations")
    st.markdown("Powered by Claude AI — generates personalised retention strategies based on customer risk profile.")

    if "last_profile" not in st.session_state:
        st.info("👈 Go to **Individual Prediction** first to generate a customer profile, then return here for AI recommendations.")
    else:
        profile = st.session_state["last_profile"]
        prob = st.session_state["last_prob"]

        st.markdown(f"**Customer Profile:** Age {profile['age']}, {profile['region']}, "
                    f"Tenure {profile['tenure_months']}m, Churn Prob: {prob:.1%}")

        if st.button("🧠 Generate AI Insights", use_container_width=True):
            with st.spinner("Generating personalised retention strategy..."):
                insights = generate_ai_insights(profile, prob)

            st.success("Insights generated successfully!")

            st.subheader("📋 Risk Summary")
            st.write(insights["risk_summary"])

            st.subheader("🎯 Retention Recommendations")
            for i, rec in enumerate(insights["recommendations"], 1):
                with st.expander(f"Recommendation {i}: {rec['action']}", expanded=True):
                    st.write(f"**Rationale:** {rec['rationale']}")
                    st.write(f"**Expected Impact:** {rec['expected_impact']}")

            st.metric("Estimated Retention Lift",
                      f"+{insights['estimated_retention_lift_pct']}%",
                      help="Estimated improvement if all recommendations are implemented")


# ══════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📈 Model Performance Comparison")

    for name, m in metrics.items():
        with st.expander(f"**{name}** — ROC-AUC: {m['roc_auc']:.4f}", expanded=(name == max(metrics, key=lambda k: metrics[k]["roc_auc"]))):
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            c2.metric("PR-AUC", f"{m['pr_auc']:.4f}")
            c3.metric("CV Score", f"{m['cv_mean']:.4f} ± {m['cv_std']:.4f}")

            if "report" in m:
                st.markdown("**Classification Report:**")
                report_df = pd.DataFrame(m["report"]).T
                st.dataframe(report_df.round(3), use_container_width=True)

            if "confusion" in m:
                cm = np.array(m["confusion"])
                fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                x=["Retained", "Churned"], y=["Retained", "Churned"],
                                color_continuous_scale="Blues")
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Megha Sharma**")
st.sidebar.markdown("MSc Business Analytics | Swansea University")
