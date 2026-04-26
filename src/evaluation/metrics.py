import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def calculate_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def evaluate_classifier(model, X_test, y_test):
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_infer

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "fpr": calculate_fpr(y_test, y_pred),
        "inference_time": inference_time,
    }

    if y_score is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_score)
    else:
        results["roc_auc"] = np.nan

    return results