# DS605: Fundamentals of Machine Learning — Lab Assignment 3

## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Piprotar Hemangi
**Student ID:** 202618059

---

## Dataset

- **Name:** Hotel Booking Demand
- **Source:** [Kaggle – Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **File used:** `hotel_bookings.csv`
- **Target variable:** `is_canceled` (binary classification — 0 = not canceled, 1 = canceled)

---

## Project Structure

```
├── README.md
├── 202618059_Lab03.ipynb
├── hotel_bookings_cleaned.csv
├── confusion_matrices.png
└── outlier_boxplot.png


---

## Part A – Data Loading and Preprocessing

### Task 1 – Data Understanding
- Loaded `hotel_bookings.csv` and inspected it using `head()`, `shape`, `info()`, `describe()`, and `dtypes`.
- Checked class distribution of `is_canceled` (target `y`).
- Separated remaining columns into `X`, and identified numerical vs categorical feature types.

### Task 2 – Missing Values, Leakage, and Outliers
- Computed missing-value count and percentage per column.
- **Dropped `company`** — 94% missing values. Imputing this column (even with KNN) would fabricate values for nearly the entire dataset and add no reliable signal.
- Retained `agent` (~13.7% missing) but treated it as a categorical/ID-like feature rather than a continuous numeric one, since it represents booking agent codes, not a measurable quantity.
- Removed `reservation_status` and `reservation_status_date` — these directly leak the final booking outcome (target leakage), since they are recorded *after* the booking is resolved.
- Checked numerical features (e.g. `lead_time`, `adr`) for outliers using boxplots and the IQR method.
- Removed extreme/clearly invalid outliers (e.g. `adr` negative or implausibly high values). **[X] rows removed** — *(update with your actual count)*.

### Task 3 – Preprocessing Pipelines
- Split data using `train_test_split(test_size=0.2, stratify=y, random_state=42)` — same split reused for all four experiments.
- **Numerical features:** `KNNImputer(n_neighbors=5)` for missing values.
- **Categorical features:** `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
- **Pipeline A:** KNNImputer + `StandardScaler` for numerical features.
- **Pipeline B:** KNNImputer + `MinMaxScaler` for numerical features.
- Built with `ColumnTransformer` + `Pipeline`, fitted only on training data to avoid leakage.

---

## Part B – Model Training and Evaluation

### Task 4 – Models Trained
Four model–pipeline combinations were trained:
1. Logistic Regression (`max_iter=1000`) + Pipeline A (StandardScaler)
2. Logistic Regression (`max_iter=1000`) + Pipeline B (MinMaxScaler)
3. Decision Tree (`random_state=42`) + Pipeline A (StandardScaler)
4. Decision Tree (`random_state=42`) + Pipeline B (MinMaxScaler)

### Task 5 – Results

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression + StandardScaler | 0.8191 | 0.8154 | 0.8031 | 0.6647 | 0.7274 |
| Logistic Regression + MinMaxScaler   | 0.8151 | 0.8121 | 0.8018 | 0.6546 | 0.7208 |
| Decision Tree + StandardScaler       | 0.9963 | 0.8592 | 0.8066 | 0.8155 | 0.8110 |
| Decision Tree + MinMaxScaler         | 0.9963 | 0.8595 | 0.8070 | 0.8158 | 0.8114 |

Confusion matrices for the best Logistic Regression (StandardScaler) and best Decision Tree (MinMaxScaler) are included in `confusion_matrices.png`.

**Overfitting:**
- Logistic Regression shows a negligible train–test gap (<1%), indicating good generalization.
- Decision Tree shows a large train–test gap (~13.7%), a clear sign of overfitting — expected for an unconstrained tree with no `max_depth` or `min_samples_leaf` limits.

---

## Task 6 – Final Observations

1. **Best overall combination:** Decision Tree + MinMaxScaler gives the best overall result (Test Accuracy ≈ 0.8595, F1 ≈ 0.8114), narrowly ahead of Decision Tree + StandardScaler, and clearly ahead of both Logistic Regression variants across all metrics.
2. **Scaler effect on Logistic Regression:** StandardScaler slightly outperforms MinMaxScaler (Test Accuracy 0.8154 vs 0.8121, F1 0.7274 vs 0.7208), consistent with Logistic Regression's gradient-based optimizer converging better on standardized features.
3. **Scaler effect on Decision Tree:** Scaling makes almost no difference (Test Accuracy 0.8592 vs 0.8595) — trees split on thresholds and are inherently scale-invariant.
4. **Overfitting:** Logistic Regression generalizes well (train/test gap < 1%), while Decision Tree overfits significantly (train/test gap ≈ 13.7%) due to being trained without depth/leaf-size constraints.
5. **Precision vs Recall trade-off:** Decision Tree achieves noticeably higher Recall than Logistic Regression (~0.815 vs ~0.65–0.66), catching more actual cancellations — useful for a hotel managing cancellation risk, despite its overfitting tendency.

---
