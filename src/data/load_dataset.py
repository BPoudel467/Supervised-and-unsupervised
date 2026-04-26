from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.utils.common import INTERIM_DIR, NIDD_RAW_DIR, UNSW_RAW_DIR, ensure_directories


def _load_any_csvs(folder: Path) -> pd.DataFrame | None:
    if not folder.exists():
        return None
    csvs = sorted([p for p in folder.glob("*.csv") if p.is_file()])
    if not csvs:
        return None
    dfs = [pd.read_csv(p) for p in csvs]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels into:
    - label: {0,1}
    - attack_cat: optional category string (kept for analysis, dropped from features)
    """
    df = _standardize_columns(df)

    # Common variants
    if "label" not in df.columns:
        for cand in ["class", "target", "y"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "label"})
                break

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a binary label column (expected `label`).")

    # Map label to 0/1 if needed
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(
            {
                "0": 0,
                "1": 1,
                "benign": 0,
                "normal": 0,
                "attack": 1,
                "malicious": 1,
                "intrusion": 1,
            }
        )
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).fillna("Unknown")

    return df


def _make_synthetic_5g_nidd(n_rows: int = 8_000, n_features: int = 49, seed: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=max(10, n_features // 3),
        n_redundant=max(5, n_features // 6),
        n_clusters_per_class=2,
        weights=[0.85, 0.15],
        flip_y=0.01,
        class_sep=1.2,
        random_state=seed,
    )

    cols = [f"f{i:02d}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y.astype(int)

    rng = np.random.default_rng(seed)
    cats = np.array(["Benign", "DoS", "DDoS", "PortScan", "Botnet", "Injection"])
    df["attack_cat"] = np.where(df["label"].to_numpy() == 0, "Benign", rng.choice(cats[1:], size=len(df)))
    return df


def load_dataset() -> pd.DataFrame:
    """
    Loads dataset from disk (prefers 5G-NIDD folder), otherwise generates a synthetic dataset
    so the pipeline can still produce the README outputs.

    Expected locations:
    - data/raw/5g-nidd/*.csv  (preferred)
    - data/raw/unsw_nb15/*.csv (fallback)
    """
    ensure_directories()

    df = _load_any_csvs(NIDD_RAW_DIR)
    source = f"{NIDD_RAW_DIR}/*.csv"

    if df is None:
        df = _load_any_csvs(UNSW_RAW_DIR)
        source = f"{UNSW_RAW_DIR}/*.csv"

    if df is None:
        df = _make_synthetic_5g_nidd()
        source = "synthetic (generated)"

    df = _ensure_label_columns(df)

    out = INTERIM_DIR / "dataset_combined.csv"
    df.to_csv(out, index=False)
    print(f"Loaded dataset from: {source}")
    print(f"Combined dataset saved to: {out}")
    print(f"Combined shape: {df.shape}")
    return df


if __name__ == "__main__":
    load_dataset()

