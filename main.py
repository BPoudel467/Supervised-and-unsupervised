from src.utils.common import ensure_directories, set_seed
from src.data.load_dataset import load_dataset
from src.data.clean_data import clean_unsw_data
from src.data.preprocess import preprocess_and_save
from src.data.imbalance import apply_smote
from src.Features.feature_selection import select_features_with_rf
from src.models.train_supervised import run_supervised_training
from src.models.train_unsupervised import run_unsupervised
from src.evaluation.compare import combine_results
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_performance_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_training_time_vs_accuracy,
)


def main():
    ensure_directories()
    set_seed()

    print("Step 1: Loading and combining dataset files...")
    df = load_dataset()

    print("Step 2: Cleaning dataset...")
    df_clean = clean_unsw_data(df)

    print("Step 3: Preprocessing and splitting...")
    preprocess_and_save(df_clean)

    print("Step 4: Handling class imbalance...")
    apply_smote()

    print("Step 5: Feature selection...")
    select_features_with_rf()

    print("Step 6: Training supervised models...")
    run_supervised_training()

    print("Step 7: Training unsupervised models...")
    run_unsupervised()

    print("Step 8: Combining results tables...")
    combine_results()

    print("Step 9: Generating figures...")
    plot_confusion_matrix()
    plot_roc_curves()
    plot_precision_recall_curves()
    plot_feature_importance()
    plot_model_performance_comparison()
    plot_training_time_vs_accuracy()

    print("Pipeline completed successfully. Outputs saved under `outputs/`.")


if __name__ == "__main__":
    main()
    