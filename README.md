# ML Assignment 2 - Classification Analysis

## 1\. Problem Statement

To build and deploy a machine learning classification application that predicts Breast Cancer diagnosis (Malignant/Benign) using multiple ML algorithms. The goal is to compare the performance of various classifiers and deploy the best-performing solution via a Streamlit web interface.

## 2\. Dataset Description

*   **Dataset:** Breast Cancer Wisconsin (Diagnostic)
*   **Source:** Scikit-Learn (UCI Machine Learning Repository)
*   **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
*   **Instances:** 569 samples
*   **Target:** Binary Classification (Malignant vs. Benign)

## 3\. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.97 | 1.00 | 0.97 | 0.99 | 0.98 | 0.94 |
| Decision Tree | 0.93 | 0.93 | 0.94 | 0.94 | 0.94 | 0.85 |
| KNN | 0.95 | 0.98 | 0.96 | 0.96 | 0.96 | 0.89 |
| Naive Bayes | 0.97 | 1.00 | 0.96 | 1.00 | 0.98 | 0.94 |
| Random Forest (Ensemble) | 0.96 | 1.00 | 0.96 | 0.99 | 0.97 | 0.93 |
| XGBoost (Ensemble) | 0.96 | 0.99 | 0.96 | 0.97 | 0.97 | 0.91 |

## 4\. Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | Excellent performance (97% Accuracy) and high MCC (0.94), proving the dataset is well-suited for linear boundaries. |
| **Decision Tree** | Lowest MCC score (0.85) among all models, likely due to overfitting on the training data compared to ensemble methods. |
| **KNN** | Performed reliably (95% Accuracy) with balanced Precision and Recall, benefiting from the feature scaling applied. |
| **Naive Bayes** | **Best Performer.** Achieved a perfect **Recall of 1.00** (100%), meaning it correctly identified every single malignant case in the test set. |
| **Random Forest (Ensemble)** | Very robust (96% Accuracy) with near-perfect AUC (0.995), reducing the variance seen in the single Decision Tree. |
| **XGBoost (Ensemble)** | Consistent high performance across all metrics (96% Accuracy), confirming its effectiveness for tabular data classification. |