"""Simple Customer Churn Prediction project for an AI internship."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data" / "customer_churn.csv"
FEATURES = ["age", "monthly_charge", "tenure_months", "contract_type", "has_internet"]


def train_model():
    """Load the data and train a beginner-friendly logistic regression model."""
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["churn"]

    numeric_columns = ["age", "monthly_charge", "tenure_months"]
    category_columns = ["contract_type", "has_internet"]
    preprocessor = ColumnTransformer(
        [
            ("numbers", "passthrough", numeric_columns),
            ("categories", OneHotEncoder(handle_unknown="ignore"), category_columns),
        ]
    )
    model = Pipeline(
        [("prepare_data", preprocessor), ("model", LogisticRegression(max_iter=1000))]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, accuracy_score(y_test, predictions), y_test, predictions


if __name__ == "__main__":
    model, accuracy, actual, predicted = train_model()
    print("Customer Churn Prediction")
    print("=" * 28)
    print(f"Test accuracy: {accuracy:.0%}")
    print("\nClassification report:")
    print(classification_report(actual, predicted, zero_division=0))

    example_customer = pd.DataFrame(
        [[26, 80, 3, "Monthly", "Yes"]], columns=FEATURES
    )
    result = model.predict(example_customer)[0]
    probability = model.predict_proba(example_customer)[0].max()
    print(f"Example customer prediction: {result} ({probability:.0%} confidence)")
