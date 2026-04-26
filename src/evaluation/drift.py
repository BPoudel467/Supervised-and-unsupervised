import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from src.utils.common import PROCESSED_DIR, MODELS_DIR


def simulate_drift():

    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    rf = joblib.load(MODELS_DIR / "randomforest.pkl")

    base_preds = rf.predict(X_test)

    base_acc = accuracy_score(y_test, base_preds)

    drifted = X_test + np.random.normal(0, 0.1, X_test.shape)

    drift_preds = rf.predict(drifted)

    drift_acc = accuracy_score(y_test, drift_preds)

    print("\nConcept Drift Simulation\n")

    print("Baseline Accuracy:", base_acc)
    print("Drift Accuracy:", drift_acc)