import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "credit_scoring_model.pkl"


model = joblib.load(MODEL_PATH)


def predict_credit_risk(customer_data):
    """
    Predict credit risk for a customer.

    Returns:
        risk: Good Credit / Bad Credit
        probability: probability of Good Credit
    """

    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0, 1]

    if prediction == 1:
        risk = "Good Credit"
    else:
        risk = "Bad Credit"

    return risk, probability