
# Lab 3: Scikit-learn Data Preprocessing and Model Performance Evaluation

**Name:** Piprotar Hemangi

**Student ID:** 202618059

**Dataset Link:** [Kaggle Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## Preprocessing Choices

* **Target & Data Leakage Removal:** The target variable selected is `is_canceled`. Columns such as `reservation_status` and `reservation_status_date` were dropped because they directly leak the final booking outcome.


* **High Missingness Removal:** The `company` column was dropped due to missing over 94% of its values.


* **Outlier Removal:** Cleaned extreme numerical anomalies in features like `adr` (e.g., negative values and values exceeding $5,000) using a strict IQR threshold.


* **Imputation:** Applied `KNNImputer(n_neighbors=5)` for missing numerical features and `SimpleImputer(strategy="most_frequent")` for categorical features.


* **Feature Scaling:** Evaluated two numerical scaling strategies within Scikit-learn pipelines: Pipeline A uses `StandardScaler()` while Pipeline B uses `MinMaxScaler()`.


* **Categorical Encoding:** Applied `OneHotEncoder(handle_unknown="ignore")` to convert categorical variables into binary vectors.


* **Data Splitting:** Data was split into 80% training and 20% testing sets using stratified sampling on `is_canceled` (`random_state=42`) to maintain class balance.



---

## Final Observations

1. **Best Overall Result:** The Decision Tree classifier paired with either pipeline provided the best overall performance, achieving a higher testing accuracy and F1-score compared to Logistic Regression.


2. **StandardScaler vs. MinMaxScaler on Logistic Regression:** Scaling significantly impacts Logistic Regression performance compared to unscaled data; however, `StandardScaler` slightly outperformed `MinMaxScaler` in convergence and overall accuracy.


3. **Effect of Scaling on Decision Tree:** Feature scaling made no meaningful difference to the Decision Tree performance, as tree-based models split nodes based on feature value order rather than feature magnitude.


4. **Overfitting Analysis:** The unconstrained Decision Tree model exhibited noticeable overfitting, achieving nearly 100% accuracy on the training set compared to a lower testing score. In contrast, Logistic Regression demonstrated minimal variance between training and testing metrics.
