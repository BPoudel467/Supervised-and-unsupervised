import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils.common import PROCESSED_DIR, MODELS_DIR, TABLES_DIR


def load_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train_selected.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    return X_train, X_test, y_test


def _calculate_fpr(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


# -------------------------
# KMEANS CLUSTERING
# -------------------------

def run_kmeans(X_train, X_test, y_test):

    start_train = time.perf_counter()
    model = KMeans(n_clusters=2, random_state=42, n_init="auto")
    model.fit(X_train)
    train_time = time.perf_counter() - start_train

    start_infer = time.perf_counter()
    # Use distance-to-centroid as anomaly score
    centers = model.cluster_centers_
    labels = model.predict(X_test)
    dists = np.linalg.norm(X_test - centers[labels], axis=1)
    inference_time = time.perf_counter() - start_infer

    # Convert to anomaly prediction using high-distance threshold
    threshold = np.percentile(dists, 95)
    preds = (dists > threshold).astype(int)

    results = {
        "model": "k-Means",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, dists),
        "fpr": _calculate_fpr(y_test, preds),
        "training_time": train_time,
        "inference_time": inference_time,
    }

    joblib.dump(model, MODELS_DIR / "kmeans.pkl")

    return results


# -------------------------
# AUTOENCODER
# -------------------------

def run_autoencoder(X_train, X_test, y_test):
    # Use a lightweight sklearn "autoencoder-like" regressor to avoid TensorFlow
    # runtime crashes on some Windows setups.
    start_train = time.perf_counter()
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    autoencoder.fit(X_train, X_train)
    train_time = time.perf_counter() - start_train

    start_infer = time.perf_counter()
    reconstructions = autoencoder.predict(X_test)
    inference_time = time.perf_counter() - start_infer

    errors = np.mean(np.square(X_test - reconstructions), axis=1)

    threshold = np.percentile(errors, 95)

    preds = (errors > threshold).astype(int)

    results = {
        "model": "Autoencoder",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, errors),
        "fpr": _calculate_fpr(y_test, preds),
        "training_time": train_time,
        "inference_time": inference_time,
    }

    # Saved with .h5 extension to match README artifact name.
    joblib.dump(autoencoder, MODELS_DIR / "autoencoder.h5")

    return results


def run_unsupervised():

    X_train, X_test, y_test = load_data()

    results = []

    results.append(run_kmeans(X_train, X_test, y_test))
    results.append(run_autoencoder(X_train, X_test, y_test))

    df = pd.DataFrame(results)

    df.to_csv(TABLES_DIR / "unsupervised_results.csv", index=False)

    print(df)


if __name__ == "__main__":
    run_unsupervised()