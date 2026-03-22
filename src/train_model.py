"""
Customer Churn Prediction Model
================================
Trains and evaluates multiple classifiers on the Enterprise Mobility dataset.
Best model is saved for deployment in the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import warnings
warnings.filterwarnings("ignore")


def load_and_preprocess(path: str = "data/customers.csv") -> tuple:
    df = pd.read_csv(path)

    # Feature engineering
    df["spend_per_rental"] = df["monthly_spend_gbp"] / (df["num_rentals_12m"] + 1)
    df["satisfaction_avg"] = (df["pricing_satisfaction"] + df["service_satisfaction"]) / 2
    df["is_young"] = (df["age"] < 26).astype(int)
    df["high_complaints"] = (df["complaints_last_6m"] >= 2).astype(int)
    df["low_engagement"] = (df["digital_engagement_score"] < 0.2).astype(int)
    df["tenure_bin"] = pd.cut(df["tenure_months"], bins=[0, 6, 12, 24, 60, 120],
                              labels=["0-6m", "6-12m", "1-2y", "2-5y", "5y+"]).astype(str)

    # Encode categoricals
    le_gender = LabelEncoder()
    le_region = LabelEncoder()
    le_tenure = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["region_enc"] = le_region.fit_transform(df["region"])
    df["tenure_bin_enc"] = le_tenure.fit_transform(df["tenure_bin"])

    feature_cols = [
        "age", "tenure_months", "monthly_spend_gbp", "num_rentals_12m",
        "digital_engagement_score", "pricing_satisfaction", "service_satisfaction",
        "nps_score", "loyalty_member", "complaints_last_6m",
        "spend_per_rental", "satisfaction_avg", "is_young",
        "high_complaints", "low_engagement",
        "gender_enc", "region_enc", "tenure_bin_enc"
    ]

    X = df[feature_cols]
    y = df["churned"]
    return X, y, feature_cols, df


def train_models(X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        data = X_train_sc if "Logistic" in name else X_train
        test_data = X_test_sc if "Logistic" in name else X_test

        model.fit(data, y_train)
        y_pred = model.predict(test_data)
        y_prob = model.predict_proba(test_data)[:, 1]

        cv_scores = cross_val_score(model, data, y_train, cv=cv, scoring="roc_auc")
        roc = roc_auc_score(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)

        results[name] = {
            "model": model,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion": confusion_matrix(y_test, y_pred).tolist(),
        }
        print(f"{name:25s} | ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results, scaler, X_test, y_test


def save_best(results, scaler, feature_cols):
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC: {best['roc_auc']:.4f})")

    out = Path("src/artifacts")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metrics = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(out / "features.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"Saved artifacts to {out}/")
    return best_name


def get_feature_importance(results, feature_cols):
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    model = results[best_name]["model"]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None

    fi = pd.DataFrame({"feature": feature_cols, "importance": imp})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nTop 10 Feature Importances:")
    print(fi.head(10).to_string(index=False))
    return fi


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("=" * 60)

    X, y, feature_cols, df = load_and_preprocess()
    print(f"\nDataset: {len(df)} records | Features: {len(feature_cols)} | Churn rate: {y.mean():.1%}\n")

    results, scaler, X_test, y_test = train_models(X, y)
    save_best(results, scaler, feature_cols)
    get_feature_importance(results, feature_cols)
    print("\nDone.")


if __name__ == "__main__":
    main()
