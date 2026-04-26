import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
)

from src.utils.common import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, TABLES_DIR


def plot_confusion_matrix():

    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    model = joblib.load(MODELS_DIR / "randomforest.pkl")

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Random Forest Confusion Matrix")

    plt.savefig(FIGURES_DIR / "confusion_matrix_rf.png")

    plt.close()


def _load_test() -> tuple[np.ndarray, np.ndarray]:
    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")
    # y_test can be a pandas Series
    y_test = np.asarray(y_test).astype(int)
    return X_test, y_test


def plot_roc_curves() -> None:
    X_test, y_test = _load_test()
    plt.figure(figsize=(7, 5))

    for name, path in [
        ("Random Forest", MODELS_DIR / "randomforest.pkl"),
        ("SVM", MODELS_DIR / "svm.pkl"),
        ("XGBoost", MODELS_DIR / "xgboost.pkl"),
    ]:
        if not path.exists():
            continue
        model = joblib.load(path)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue
        RocCurveDisplay.from_predictions(y_test, y_score, name=name)

    plt.title("ROC Curve (Supervised Models)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_ids.png")
    plt.close()


def plot_precision_recall_curves() -> None:
    X_test, y_test = _load_test()
    plt.figure(figsize=(7, 5))

    for name, path in [
        ("Random Forest", MODELS_DIR / "randomforest.pkl"),
        ("SVM", MODELS_DIR / "svm.pkl"),
        ("XGBoost", MODELS_DIR / "xgboost.pkl"),
    ]:
        if not path.exists():
            continue
        model = joblib.load(path)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.plot(recall, precision, label=name)

    plt.title("Precision-Recall Curve Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png")
    plt.close()


def plot_feature_importance(top_n: int = 20) -> None:
    selector_path = PROCESSED_DIR / "feature_selector.pkl"
    preprocessor_path = PROCESSED_DIR / "preprocessor.pkl"
    meta_path = PROCESSED_DIR / "metadata.pkl"

    if not selector_path.exists() or not preprocessor_path.exists() or not meta_path.exists():
        return

    selector = joblib.load(selector_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(meta_path)

    # Rebuild feature names post-preprocess (num + onehot cat)
    num_cols = metadata.get("numerical_cols", [])
    cat_cols = metadata.get("categorical_cols", [])

    feature_names: list[str] = []
    feature_names.extend([str(c) for c in num_cols])
    if cat_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"]
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            ohe_names = [f"{c}_encoded" for c in cat_cols]
        feature_names.extend(ohe_names)

    if not hasattr(selector, "estimator_") and hasattr(selector, "estimator"):
        est = selector.estimator
    else:
        est = getattr(selector, "estimator_", None) or getattr(selector, "estimator", None)

    importances = None
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)

    if importances is None or len(importances) != len(feature_names):
        return

    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
    plt.title("Feature Importance (Random Forest Selector)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_ids.png")
    plt.close()


def plot_model_performance_comparison() -> None:
    path = TABLES_DIR / "model_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    metrics = ["accuracy", "precision", "recall", "f1"]
    for m in metrics:
        if m not in df.columns:
            return

    models = df["model"].astype(str).tolist()
    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + i * width, df[m].astype(float), width=width, label=m.capitalize() if m != "f1" else "F1 Score")

    plt.xticks(x + width * 1.5, models, rotation=15)
    plt.ylim(0, 1.0)
    plt.title("Performance Comparison of IDS Models")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_performance_comparison.png")
    plt.close()


def plot_training_time_vs_accuracy() -> None:
    path = TABLES_DIR / "model_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "training_time" not in df.columns or "accuracy" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    x = df["training_time"].astype(float).to_numpy()
    y = df["accuracy"].astype(float).to_numpy()
    plt.scatter(x, y)
    for i, name in enumerate(df["model"].astype(str).tolist()):
        plt.annotate(name, (x[i], y[i]))

    plt.title("Training Time vs Accuracy of IDS Models")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_time_vs_accuracy.png")
    plt.close()