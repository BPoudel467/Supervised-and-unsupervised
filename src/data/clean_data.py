import pandas as pd
from src.utils.common import INTERIM_DIR

DROP_COLUMNS_IF_PRESENT = [
    "id",  # common in UNSW variants
]


def clean_unsw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed duplicates: {before - after}")

    # Drop optional columns
    drop_cols = [col for col in DROP_COLUMNS_IF_PRESENT if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    # Save cleaned copy
    output_path = INTERIM_DIR / "Combined1.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Cleaned shape: {df.shape}")

    return df


def load_and_clean() -> pd.DataFrame:
    input_path = INTERIM_DIR / "unsw_combined.csv"
    df = pd.read_csv(input_path)
    return clean_unsw_data(df)


if __name__ == "__main__":
    load_and_clean()
