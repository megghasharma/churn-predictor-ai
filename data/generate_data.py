"""Generate synthetic customer churn dataset (2,500 records, ~30% churn)."""
import pandas as pd, numpy as np, pathlib

def main():
    np.random.seed(42)
    N = 2500
    age_p = np.array([3.0 if a < 26 else 2.0 if a < 35 else 1.5 if a < 50 else 1.0 for a in range(18, 66)])
    age_p /= age_p.sum()

    age = np.random.choice(range(18, 66), N, p=age_p)
    gender = np.random.choice(["Male", "Female", "Other"], N, p=[0.52, 0.46, 0.02])
    tenure = np.clip(np.random.exponential(18, N), 1, 120).astype(int)
    spend = np.round(np.random.lognormal(4.2, 0.5, N), 2)
    rentals = np.random.poisson(6, N)
    digital = np.random.beta(2, 5, N).round(3)
    price_sat = np.random.randint(1, 11, N)
    svc_sat = np.random.randint(1, 11, N)
    nps = np.random.randint(-100, 101, N)
    loyal = np.random.choice([0, 1], N, p=[0.4, 0.6])
    complaints = np.random.poisson(0.8, N)
    region = np.random.choice(
        ["London","South East","South West","Midlands","North West","North East","Scotland","Wales","East Anglia"],
        N, p=[.22,.14,.10,.13,.12,.08,.09,.05,.07])

    # Tuned for ~28% churn with strong predictive signal
    logit = (
        1.0
        + 1.5 * (age < 26).astype(float)
        - 0.05 * tenure
        - 0.8 * loyal
        + 0.7 * complaints
        - 0.20 * price_sat
        - 0.12 * svc_sat
        - 4.0 * digital
        - 0.004 * nps
        + np.random.normal(0, 0.1, N)
    )
    churned = (np.random.rand(N) < 1/(1+np.exp(-logit))).astype(int)

    df = pd.DataFrame(dict(
        customer_id=[f"EM-{i:05d}" for i in range(1, N+1)],
        age=age, gender=gender, tenure_months=tenure, monthly_spend_gbp=spend,
        num_rentals_12m=rentals, digital_engagement_score=digital,
        pricing_satisfaction=price_sat, service_satisfaction=svc_sat,
        nps_score=nps, loyalty_member=loyal, complaints_last_6m=complaints,
        region=region, churned=churned))

    out = pathlib.Path(__file__).parent / "customers.csv"
    df.to_csv(out, index=False)
    print(f"Created {N} records | Churn rate: {churned.mean():.1%} | {out}")

if __name__ == "__main__":
    main()
