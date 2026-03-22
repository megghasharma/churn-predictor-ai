# 🔮 AI-Powered Customer Churn Predictor

> Predicting and preventing customer churn using Machine Learning + Claude AI for automated retention strategy generation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Claude AI](https://img.shields.io/badge/Claude_AI-Powered-blueviolet)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## 📌 Overview

This project builds an end-to-end customer churn prediction system for a UK car rental company (modelled on Enterprise Mobility). It combines:

- **Machine Learning** — Logistic Regression, Random Forest, and Gradient Boosting classifiers trained on 2,500 customer records to predict churn risk (best model ROC-AUC 0.816)
- **AI-Powered Insights** — Integrates the Anthropic Claude API to generate personalised, natural-language retention recommendations for at-risk customers
- **Interactive Dashboard** — Streamlit app with regional churn heatmaps, demographic analysis, individual predictions, and AI-generated action plans

## 🧠 Key Findings

| Insight | Detail |
|---------|--------|
| **Age** | 18-25 year-olds show ~25% higher churn vs other groups |
| **Digital Engagement** | Customers with engagement score < 0.20 are 3x more likely to churn |
| **Pricing Satisfaction** | Scores below 5/10 are the #2 predictor of churn |
| **Loyalty Members** | Churn at roughly half the rate of non-members |
| **Complaints** | 2+ complaints in 6 months raises churn probability by ~40% |

## 🏗️ Architecture

```
├── data/
│   ├── generate_data.py      # Synthetic data generator (2,500 records)
│   └── customers.csv          # Generated dataset
├── src/
│   ├── train_model.py         # ML pipeline: preprocessing, training, evaluation
│   ├── ai_insights.py         # Claude API integration for retention recommendations
│   ├── app.py                 # Streamlit dashboard (4 pages)
│   └── artifacts/             # Saved model, scaler, metrics
├── notebooks/
│   └── EDA.ipynb              # Exploratory data analysis
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/churn-predictor-ai.git
cd churn-predictor-ai
pip install -r requirements.txt

# Generate data and train model
python data/generate_data.py
python src/train_model.py

# Launch dashboard
streamlit run src/app.py
```

### Enable AI-Powered Insights (Optional)

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Without an API key, the app falls back to a rule-based recommendation engine. With Claude AI enabled, recommendations are contextually richer and tailored to each customer's specific risk profile.

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC | CV Score |
|-------|---------|--------|----------|
| **Logistic Regression** | **0.816** | **0.438** | **0.816 ± 0.017** |
| Random Forest | 0.800 | 0.466 | 0.773 ± 0.011 |
| Gradient Boosting | 0.791 | 0.475 | 0.750 ± 0.033 |

## 🤖 AI Integration

The Claude API generates structured JSON responses containing:
- **Risk summary** — plain-English explanation of why the customer is at risk
- **3 ranked recommendations** — specific actions with rationale and expected impact
- **Retention lift estimate** — projected improvement percentage

Example output for a high-risk young customer:
```json
{
  "risk_summary": "This customer has a HIGH churn risk (78%). Primary drivers are young age (22), very low digital engagement (0.12), and poor pricing satisfaction (4/10).",
  "recommendations": [
    {
      "action": "Offer student/young driver discount with gamified loyalty tiers",
      "rationale": "18-25 age group shows 25% higher churn driven by price sensitivity",
      "expected_impact": "High"
    }
  ],
  "estimated_retention_lift_pct": 22
}
```

## 🛠️ Tech Stack

**Data & ML:** Python, Pandas, NumPy, scikit-learn
**Visualisation:** Plotly, Streamlit
**AI:** Anthropic Claude API (claude-sonnet-4-20250514)
**Deployment:** Streamlit Cloud / Docker

## 👩‍💻 Author

**Megha Sharma** — MSc Management (Business Analytics), Swansea University
- [LinkedIn](https://linkedin.com/in/megha-sharma-6a47861a9)
- Built as part of portfolio demonstrating ML + AI integration for business analytics

## 📄 Licence

MIT
