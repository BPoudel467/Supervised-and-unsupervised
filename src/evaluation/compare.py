import pandas as pd
from src.utils.common import TABLES_DIR


def combine_results():

    supervised = pd.read_csv(TABLES_DIR / "supervised_results.csv")
    unsupervised = pd.read_csv(TABLES_DIR / "unsupervised_results.csv")

    combined = pd.concat([supervised, unsupervised])

    combined.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    print("\nFinal Model Comparison Table\n")
    print(combined)


if __name__ == "__main__":
    combine_results()