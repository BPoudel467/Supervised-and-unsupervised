import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.common import PROCESSED_DIR, RANDOM_STATE


TARGET_COL = "label"
ATTACK_CAT_COL = "attack_cat"


def split_data(df: pd.DataFrame):
    df = df.copy()

    df.columns = [c.strip().lower() for c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # Keep attack_cat only for analysis, not as model feature
    feature_drop_cols = [TARGET_COL]
    if ATTACK_CAT_COL in df.columns:
        feature_drop_cols.append(ATTACK_CAT_COL)

    X = df.drop(columns=feature_drop_cols)
    y = df[TARGET_COL]

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train: pd.DataFrame):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor, numerical_cols, categorical_cols


def preprocess_and_save(df: pd.DataFrame):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Save arrays
    joblib.dump(X_train_processed, PROCESSED_DIR / "X_train.pkl")
    joblib.dump(X_val_processed, PROCESSED_DIR / "X_val.pkl")
    joblib.dump(X_test_processed, PROCESSED_DIR / "X_test.pkl")

    joblib.dump(y_train, PROCESSED_DIR / "y_train.pkl")
    joblib.dump(y_val, PROCESSED_DIR / "y_val.pkl")
    joblib.dump(y_test, PROCESSED_DIR / "y_test.pkl")

    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.pkl")

    metadata = {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
    joblib.dump(metadata, PROCESSED_DIR / "metadata.pkl")

    print("Preprocessed data saved to data/processed/")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_val_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")


if __name__ == "__main__":
    df = pd.read_csv("data/interim/5G-NIDD_cleaned.csv")
    preprocess_and_save(df)
    