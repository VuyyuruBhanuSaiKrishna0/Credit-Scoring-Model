"""Demo credit scoring classifier using synthetic data. Not for real lending decisions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def make_demo_data(n=2500, seed=42):
    """Create clearly synthetic demonstration data; replace with an authorized real dataset for real analysis."""
    rng = np.random.default_rng(seed)
    income = rng.integers(18000, 150001, n)
    debt = rng.integers(0, 80001, n)
    age = rng.integers(18, 76, n)
    late_payments = rng.integers(0, 8, n)
    credit_history_years = rng.integers(0, 31, n)
    employment = rng.choice(["employed", "self-employed", "unemployed"], n, p=[.68, .20, .12])
    # Synthetic label only for demonstrating an end-to-end ML workflow.
    score = (income / 18000) - (debt / 14000) - (late_payments * 1.25) + (credit_history_years * .18) + (age * .025)
    score += np.where(employment == "unemployed", -2.0, 0)
    probability = 1 / (1 + np.exp(-score / 2.7))
    approved = rng.binomial(1, probability)
    return pd.DataFrame({
        "income": income, "debt": debt, "age": age,
        "late_payments": late_payments, "credit_history_years": credit_history_years,
        "employment": employment, "creditworthy": approved
    })

def main():
    df = make_demo_data()
    X, y = df.drop(columns="creditworthy"), df["creditworthy"]
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    }
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
        print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")
        print(classification_report(y_test, pred, target_names=["Not creditworthy (demo)", "Creditworthy (demo)"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nNote: labels are synthetic and this project is educational, not for actual credit decisions.")

if __name__ == "__main__":
    main()
