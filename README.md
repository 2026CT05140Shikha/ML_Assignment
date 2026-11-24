# Hotel Booking Cancellation Prediction

## 1. Problem Statement
Hotel booking cancellations represent a significant challenge in the hospitality industry, leading to revenue loss, suboptimal inventory management, and inaccurate demand forecasting. The objective of this project is to build, evaluate, and compare multiple supervised machine learning classification models to accurately predict whether a hotel booking will be canceled (`is_canceled = 1`) or honored (`is_canceled = 0`) based on historical reservation attributes.


## 2. Dataset Description and source
The dataset (`hotel_bookings.csv`) contains historical booking records for both a City Hotel and a Resort Hotel.The dataset source is from https://www.kaggle.com/datasets

* **Initial Dataset Shape:** 119,390 instances (rows) and 32 features (columns).
* **Cleaned Dataset Shape:** 87,396 instances (after removing 31,994 duplicate records) and 28 columns (27 predictor features + 1 target variable).
* **Target Variable (`is_canceled`):** Binary classification target (`0` = Not Canceled [~72.51%], `1` = Canceled [~27.49%]).

### Data Preprocessing & Feature Engineering

1. **Duplicate Removal:** Removed 31,994 exact duplicate rows to prevent overfitting and inflated evaluation metrics.
2. **Data Leakage & High-Sparsity Removal:** Dropped `reservation_status` and `reservation_status_date` (which directly leak the target outcome), as well as `company` (94% missing) and `agent` (high-cardinality identifier with ~14% missing values).
3. **Missing Value Imputation:**
   * Numerical missing values (e.g., `children`) were imputed using the **median**.
   * Categorical missing values (e.g., `country`) were imputed using the constant `'Unknown'`.
4. **Train-Test Split:** Performed an **80:20 stratified split** (`random_state=42`), yielding **69,916 training records** and **17,480 testing records** while preserving class proportions. All transformations were fitted exclusively on the training set to prevent data leakage.
5. **Encoding & Scaling:**
   * **Numerical Features (17):** Standardized using `StandardScaler`.
   * **Categorical Features (10):** Transformed using `OneHotEncoder(handle_unknown='ignore')`, expanding the final feature space to **248 features**.



## 3. Machine Learning Models Used
Five classification algorithms were implemented and evaluated

1. **Logistic Regression:** A linear baseline classifier configured with `max_iter=1000` and `random_state=42`.
2. **Decision Tree Classifier:** A non-linear tree-based model (`random_state=42`) that partitions feature space using Gini impurity.
3. **K-Nearest Neighbors (kNN):** A distance-based, non-parametric classifier configured with `n_neighbors=5`.
4. **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem assuming feature independence and Gaussian distributions.
5. **Random Forest Classifier (Ensemble):** An ensemble bagging method combining 100 decision trees (`n_estimators=100`, `random_state=42`) to reduce variance and improve generalization.

---

## 4. Model Comparison Table

All models were evaluated across six standard binary classification metrics on the test dataset (`17,480` samples):

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7947 | 0.8444 | 0.6773 | 0.4835 | 0.5642 | 0.4453 |
| **Decision Tree** | 0.7832 | 0.7331 | 0.6034 | 0.6171 | 0.6101 | 0.4601 |
| **kNN (n=5)** | 0.7863 | 0.8152 | 0.6190 | 0.5790 | 0.5983 | 0.4535 |
| **Gaussian Naive Bayes** | 0.3814 | 0.5710 | 0.3060 | **0.9865** | 0.4671 | 0.1945 |
| **Random Forest (Ensemble)** | **0.8402** | **0.8989** | **0.7596** | 0.6123 | **0.6780** | **0.5792** |

---

## 5. Visualization & Graph Representation

### A. Receiver Operating Characteristic (ROC) Curves
The notebook plots the ROC curves comparing the True Positive Rate (TPR) against the False Positive Rate (FPR) for all five classifiers against a random guessing baseline (`AUC = 0.5000`):
* **Random Forest** achieves the highest curve elevation (`AUC = 0.8989`), showing superior class separation across all probability thresholds.
* **Logistic Regression** (`AUC = 0.8444`) and **kNN** (`AUC = 0.8152`) demonstrate strong discriminative power.
* **Decision Tree** (`AUC = 0.7331`) exhibits a sharper, less smooth trade-off due to unpruned leaf probability estimates.
* **Gaussian Naive Bayes** (`AUC = 0.5710`) hugs closest to the diagonal random-guess line.

### B. Evaluation Metrics Comparison Chart (ASCII Bar Representation)

```text
Metric Comparison Across Models (Scale: 0.00 to 1.00)

ACCURACY
Random Forest        |██████████████████████████████████████████ 0.8402
Logistic Regression  |███████████████████████████████████████▋   0.7947
kNN                  |███████████████████████████████████████▎   0.7863
Decision Tree        |███████████████████████████████████████▏   0.7832
Gaussian Naive Bayes |███████████████████                        0.3814

AUC SCORE
Random Forest        |████████████████████████████████████████████▉ 0.8989
Logistic Regression  |██████████████████████████████████████████▏   0.8444
kNN                  |████████████████████████████████████████▊     0.8152
Decision Tree        |████████████████████████████████████▋         0.7331
Gaussian Naive Bayes |████████████████████████████▌                 0.5710

F1 SCORE
Random Forest        |█████████████████████████████████▉ 0.6780
Decision Tree        |██████████████████████████████▌    0.6101
kNN                  |█████████████████████████████▉     0.5983
Logistic Regression  |████████████████████████████▏      0.5642
Gaussian Naive Bayes |███████████████████████▎           0.4671

MCC SCORE
Random Forest        |████████████████████████████▉ 0.5792
Decision Tree        |███████████████████████       0.4601
kNN                  |██████████████████████▋       0.4535
Logistic Regression  |██████████████████████▎       0.4453
Gaussian Naive Bayes |█████████▋                    0.1945