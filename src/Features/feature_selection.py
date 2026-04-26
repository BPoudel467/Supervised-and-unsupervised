import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from src.utils.common import PROCESSED_DIR, RANDOM_STATE


def select_features_with_rf(threshold: str = "median") -> None:
    X_train = joblib.load(PROCESSED_DIR / "X_train_resampled.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train_resampled.pkl")
    X_val = joblib.load(PROCESSED_DIR / "X_val.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test.pkl")

    selector_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(selector_model, prefit=True, threshold=threshold)

    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    joblib.dump(X_train_sel, PROCESSED_DIR / "X_train_selected.pkl")
    joblib.dump(X_val_sel, PROCESSED_DIR / "X_val_selected.pkl")
    joblib.dump(X_test_sel, PROCESSED_DIR / "X_test_selected.pkl")
    joblib.dump(selector, PROCESSED_DIR / "feature_selector.pkl")

    print(f"Selected train shape: {X_train_sel.shape}")
    print(f"Selected validation shape: {X_val_sel.shape}")
    print(f"Selected test shape: {X_test_sel.shape}")


if __name__ == "__main__":
    select_features_with_rf()