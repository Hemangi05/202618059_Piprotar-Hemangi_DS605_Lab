# DS605 Lab 5 

## Dataset

**Dataset:** UCI Productivity Prediction of Garment Employees

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees

---

## Objective

The objective of this lab is to compare machine learning models implemented using:

- Scikit-learn
- NumPy and Pandas

The models used are:

- Linear Regression
- Logistic Regression

---
 
## Part A — Scikit-learn Implementation
 
Part A uses Scikit-learn for preprocessing, model training, prediction, and evaluation.
 
### Regression
 
**Target:** `actual_productivity`
 
**Model:** Linear Regression
 
**Metrics:** MAE, RMSE, R²
 
| Metric | Scikit-learn |
|---|---|
| MAE | 0.108968 |
| RMSE | 0.145976 |
| R² | 0.266219 |
| Training Time | 0.022159 s |
| Prediction Time | 0.005800 s |
 
### Classification
 
A new target, `MeetsTarget`, is created:
 
- `MeetsTarget = 1` if `actual_productivity >= targeted_productivity`
- `MeetsTarget = 0` otherwise
**Model:** Logistic Regression
 
> Note: `actual_productivity` is **not** used as an input feature for classification.
 
**Metrics:** Accuracy, Precision, Recall, F1-score
 
| Metric | Scikit-learn |
|---|---|
| Accuracy | 0.720833 |
| Precision | 0.764706 |
| Recall | 0.891429 |
| F1-score | 0.823219 |
| Training Time | 0.024506 s |
| Prediction Time | 0.004866 s |
 
---
 
## Part B — NumPy/Pandas From Scratch
 
Part B implements the full machine learning workflow manually using only NumPy and Pandas. The following steps are implemented by hand:
 
- Missing value handling
- Categorical encoding
- Feature scaling
- Linear Regression
- Logistic Regression (Sigmoid function + Gradient descent)
- Prediction
- Evaluation metrics
The same train-test split from Part A is reused for consistency.
 
### Important Requirement
 
The manual section does **not** use Scikit-learn preprocessing, models, metrics, or train-test utilities. Python timing utilities (e.g., `time`) are used only to measure execution time.
 
### Manual Linear Regression
 
| Metric | Manual |
|---|---|
| MAE | 0.109207 |
| RMSE | 0.146328 |
| R² | 0.262670 |
| Training Time | 0.097347 s |
| Prediction Time | 0.000063 s |
 
### Manual Logistic Regression
 
Implemented using:
 
- Sigmoid function
- Gradient descent
- Probability prediction
- Classification threshold
- Manual evaluation metrics
| Metric | Manual |
|---|---|
| Accuracy | 0.750000 |
| Precision | 0.786070 |
| Recall | 0.902857 |
| F1-score | 0.840426 |
| Training Time | 0.384199 s |
| Prediction Time | 0.000142 s |
 
---
 
## Part C — Optimization
 
Part C improves the manual Logistic Regression implementation using NumPy vectorization, including:
 
- Vectorized matrix operations
- Gradient descent
- Learning rate adjustment
- Efficient NumPy calculations
### Optimized Results
 
| Metric | Optimized Manual |
|---|---|
| Accuracy | 0.729167 |
| Precision | 0.775000 |
| Recall | 0.885714 |
| F1-score | 0.826667 |
| Training Time | 0.225804 s |
| Prediction Time | 0.000111 s |
 
---
 
## Comparison
 
### Linear Regression
 
| Implementation | MAE | RMSE | R² | Training Time |
|---|---|---|---|---|
| Scikit-learn | 0.108968 | 0.145976 | 0.266219 | 0.022159 s |
| Manual | 0.109207 | 0.146328 | 0.262670 | 0.097347 s |
 
### Logistic Regression
 
| Implementation | Accuracy | Precision | Recall | F1-score | Training Time |
|---|---|---|---|---|---|
| Scikit-learn | 0.720833 | 0.764706 | 0.891429 | 0.823219 | 0.024506 s |
| Manual | 0.750000 | 0.786070 | 0.902857 | 0.840426 | 0.384199 s |
| Optimized Manual | 0.729167 | 0.775000 | 0.885714 | 0.826667 | 0.225804 s |
 
---
 
## Key Observations
 
- The Linear Regression results from Scikit-learn and the manual implementation are very similar.
- Scikit-learn has a shorter training time in this experiment because its implementations are optimized.
- Logistic Regression was implemented manually using the sigmoid function and gradient descent.
- NumPy vectorization reduced the training time of the optimized manual Logistic Regression compared with the basic manual implementation in this run.
- Learning rate and number of iterations affect the performance and training time of manual Logistic Regression.
- The manual implementation requires more code because preprocessing, model training, prediction, and evaluation are implemented directly.
- The same train-test split, features, target definitions, and evaluation metrics were used to make the comparison fair and reproducible.
- The experiment shows the difference between using ready-made machine learning libraries and understanding the algorithms through manual implementation.

- ---
 
## Conclusion
 
Overall, the Scikit-learn and manual implementations produced comparable results for both Linear and Logistic Regression, validating the correctness of the from-scratch approach, while Scikit-learn remained faster due to its optimized internal routines. The optimized manual Logistic Regression (Part C) showed that NumPy vectorization can substantially reduce training time versus the basic manual version without sacrificing performance, illustrating that library implementations prioritize speed and production-readiness, whereas manual implementations are valuable for understanding how these algorithms work under the hood.
 
 

