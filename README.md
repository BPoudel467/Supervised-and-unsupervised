Paste this complete version.

# Machine Learning Based Intrusion Detection System (IDS)

## Overview

This project implements a comparative study of machine learning models for network intrusion detection using the **5G-NIDD**. The goal is to evaluate the effectiveness of supervised and unsupervised machine learning algorithms in detecting malicious network activity within enterprise environments.

The study forms part of a **Master’s Dissertation in Computer Science** focusing on the application of Artificial Intelligence in cybersecurity.

---

## Dataset
Dataset characteristics:

- Realistic modern network traffic
- Multiple attack categories
- 49 features describing network flows
- Large scale dataset suitable for machine learning research

The dataset is:


5G-NIDD


These files are combined during preprocessing to create the final dataset used for model training.

---

## Project Structure


ids-comparative-study/

data/
5G-NIDD

src/
data_preprocessing.py
feature_engineering.py
supervised_models.py
unsupervised_models.py
evaluation.py

outputs/

tables/
supervised_results.csv
unsupervised_results.csv
model_comparison.csv

figures/
confusion_matrix_rf.png
roc_curve_ids.png
feature_importance_ids.png
model_performance_comparison.png
precision_recall_curve.png
training_time_vs_accuracy.png

models/
randomforest.pkl
svm.pkl
xgboost.pkl
kmeans.pkl
autoencoder.h5

run_ids_pipeline.py

requirements.txt
README.md


---

## Machine Learning Models Implemented

### Supervised Models

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Unsupervised Models

- k-Means Clustering
- Autoencoder Neural Network

These models are evaluated to compare their effectiveness in identifying network intrusions.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records
- Handling missing values
- Encoding categorical variables
- Feature scaling using StandardScaler
- Class balancing using SMOTE
- Dataset splitting

Dataset split:


Training: 70%
Validation: 15%
Testing: 15%


---

## Feature Engineering

Feature engineering techniques used include:

- Correlation analysis
- Recursive Feature Elimination (RFE)
- Random Forest feature importance
- Optional PCA dimensionality reduction

These steps improve model performance and reduce computational complexity.

---

## Evaluation Metrics

The models are evaluated using standard IDS performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- False Positive Rate

These metrics allow comprehensive comparison of detection capability.

---


Among the models tested, **XGBoost achieved the highest detection performance**.

---

## Visualisations

The system generates multiple figures for analysis:

- Confusion Matrix
- ROC Curve
- Feature Importance
- Precision-Recall Curve
- Model Performance Comparison
- Training Time vs Accuracy

These figures are used in the dissertation results chapter.

---

## Installation

Clone the repository


cd ids-comparative-study


Install dependencies:


pip install -r requirements.txt


---

## Running the System

To run the full IDS pipeline:


python main.py


This script will:

1. Load 5G-NIDD dataset files
2. Perform preprocessing and feature engineering
3. Train machine learning models
4. Evaluate detection performance
5. Save trained models
6. Generate tables and visualisations

---

## UI Dashboard (all outputs + figures)

To browse all generated figures, tables, and saved models in one place:

- Install dependencies: `pip install -r requirements.txt`
- Start the dashboard: `python -m streamlit run ui_app.py`

The dashboard auto-discovers files under:

- `outputs/figures`
- `outputs/tables`
- `outputs/models`

## Outputs

After execution the following outputs will be generated:

### Tables


outputs/tables/

supervised_results.csv
unsupervised_results.csv
model_comparison.csv


### Figures


outputs/figures/

confusion_matrix_rf.png
roc_curve_ids.png
feature_importance_ids.png
model_performance_comparison.png
precision_recall_curve.png
training_time_vs_accuracy.png

# Extra figures (older screenshots renamed)
performance_comparison_lineplot.png
training_time_vs_accuracy_lineplot.png
roc_curve_comparative_all_models.png


### Models


outputs/models/

randomforest.pkl
svm.pkl
xgboost.pkl
kmeans.pkl
autoencoder.h5


---

## Enterprise Relevance

The results of this study provide insights into the suitability of machine learning models for **Security Operations Centres (SOC)**.

Key observations:

- XGBoost provides the best detection performance.
- Random Forest offers strong interpretability and stability.
- Autoencoders can detect novel attacks without labelled data.
- Unsupervised methods reduce reliance on labelled datasets.

These findings support the development of AI-driven intrusion detection systems in enterprise cybersecurity environments.

---

## Future Work

Potential improvements include:

- Deep learning models (CNN-LSTM IDS)
- Online learning for concept drift adaptation
- Real-time intrusion detection deployment
- Integration with SIEM platforms

---