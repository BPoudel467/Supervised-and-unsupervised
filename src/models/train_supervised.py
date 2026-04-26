import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_classifier
from src.utils.common import PROCESSED_DIR, MODELS_DIR, TABLES_DIR, RANDOM_STATE


def load_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train_selected.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train_resampled.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def train_svm(X_train, y_train):
    model = SVC(
        C=1.0,
        gamma="scale",
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def run_supervised_training():
    X_train, X_test, y_train, y_test = load_data()
    results = []

    models = {
        "RandomForest": train_random_forest,
        "SVM": train_svm,
        "XGBoost": train_xgboost,
    }

    for model_name, trainer in models.items():
        print(f"Training {model_name}...")
        model, train_time = trainer(X_train, y_train)

        metrics = evaluate_classifier(model, X_test, y_test)
        metrics["model"] = model_name
        metrics["training_time"] = train_time
        results.append(metrics)

        joblib.dump(model, MODELS_DIR / f"{model_name.lower()}.pkl")

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "training_time", "inference_time"]
    ]
    results_df.to_csv(TABLES_DIR / "supervised_results.csv", index=False)
    print(results_df)
    print("Supervised results saved.")


if __name__ == "__main__":
    run_supervised_training()