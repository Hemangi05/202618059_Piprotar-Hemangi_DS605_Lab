import pandas as pd
import numpy as np
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Manual implementation must use only NumPy and Pandas.
# Use the same split, features, targets and metrics for a fair comparison.

# Load dataset
data = pd.read_csv("data/garments_worker_productivity.csv")

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Create classification target
data["MeetsTarget"] = (
    data["actual_productivity"] >= data["targeted_productivity"]
).astype(int)

print("\nMeetsTarget:")
print(data["MeetsTarget"].value_counts())

# Remove target columns from input features
X = data.drop(
    columns=["actual_productivity", "MeetsTarget"]
)

# Targets
y_regression = data["actual_productivity"]
y_classification = data["MeetsTarget"]

# Create fixed train-test split
train_index, test_index = train_test_split(
    np.arange(len(data)),
    test_size=0.20,
    random_state=42,
    stratify=y_classification
)

# Save the same split for Part B
np.savez(
    "data/split_data.npz",
    train_index=train_index,
    test_index=test_index
)

# Create train and test data
X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_reg_train = y_regression.iloc[train_index]
y_reg_test = y_regression.iloc[test_index]

y_cls_train = y_classification.iloc[train_index]
y_cls_test = y_classification.iloc[test_index]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Find numerical and categorical columns
numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns

print("\nNumerical columns:")
print(list(numeric_columns))

print("\nCategorical columns:")
print(list(categorical_columns))

# Preprocessing
numeric_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

categorical_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_steps, numeric_columns),
    ("categorical", categorical_steps, categorical_columns)
])

# ---------------- LINEAR REGRESSION ----------------

print("\nLinear Regression")

linear_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

start = time.perf_counter()

linear_model.fit(X_train, y_reg_train)

linear_train_time = time.perf_counter() - start

start = time.perf_counter()

linear_prediction = linear_model.predict(X_test)

linear_prediction_time = time.perf_counter() - start

mae = mean_absolute_error(
    y_reg_test,
    linear_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        linear_prediction
    )
)

r2 = r2_score(
    y_reg_test,
    linear_prediction
)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Training time:", linear_train_time)
print("Prediction time:", linear_prediction_time)

# ---------------- LOGISTIC REGRESSION ----------------

print("\nLogistic Regression")

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

start = time.perf_counter()

logistic_model.fit(X_train, y_cls_train)

logistic_train_time = time.perf_counter() - start

start = time.perf_counter()

logistic_prediction = logistic_model.predict(X_test)

logistic_prediction_time = time.perf_counter() - start

accuracy = accuracy_score(
    y_cls_test,
    logistic_prediction
)

precision = precision_score(
    y_cls_test,
    logistic_prediction,
    zero_division=0
)

recall = recall_score(
    y_cls_test,
    logistic_prediction,
    zero_division=0
)

f1 = f1_score(
    y_cls_test,
    logistic_prediction,
    zero_division=0
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Training time:", logistic_train_time)
print("Prediction time:", logistic_prediction_time)

# ---------------- RESULTS TABLE ----------------

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Logistic Regression"
    ],

    "MAE": [
        mae,
        np.nan
    ],

    "RMSE": [
        rmse,
        np.nan
    ],

    "R2": [
        r2,
        np.nan
    ],

    "Accuracy": [
        np.nan,
        accuracy
    ],

    "Precision": [
        np.nan,
        precision
    ],

    "Recall": [
        np.nan,
        recall
    ],

    "F1": [
        np.nan,
        f1
    ],

    "Training Time": [
        linear_train_time,
        logistic_train_time
    ],

    "Prediction Time": [
        linear_prediction_time,
        logistic_prediction_time
    ]
})

print("\nFinal Part A Results:")
print(results)

# Save results
results.to_csv(
    "data/part_a_results.csv",
    index=False
)

results_json = results.to_dict(
    orient="records"
)

with open(
    "data/part_a_results.json",
    "w"
) as file:
    json.dump(
        results_json,
        file,
        indent=4
    )

print("\nPart A completed.")