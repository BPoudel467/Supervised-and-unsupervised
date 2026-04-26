import joblib
from imblearn.over_sampling import SMOTE
from src.utils.common import PROCESSED_DIR, RANDOM_STATE


def apply_smote():
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    joblib.dump(X_train_resampled, PROCESSED_DIR / "X_train_resampled.pkl")
    joblib.dump(y_train_resampled, PROCESSED_DIR / "y_train_resampled.pkl")

    print("SMOTE applied to training set only.")
    print(f"Original train shape: {X_train.shape}")
    print(f"Resampled train shape: {X_train_resampled.shape}")


if __name__ == "__main__":
    apply_smote()
    