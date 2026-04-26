import pandas as pd
from src.data.load_dataset import load_dataset


def load_and_combine_unsw() -> pd.DataFrame:
    """
    Backwards-compatible wrapper.

    The project now uses `src.data.load_dataset.load_dataset()` which searches common
    raw-data folders and can generate a synthetic dataset if none is present.
    """
    return load_dataset()


if __name__ == "__main__":
    load_and_combine_unsw()