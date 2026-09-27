import pandas as pd
import numpy as np
import time
import json

# Do not use Scikit-learn preprocessing, models, metrics, or train-test utilities in this section.


# Load dataset
data = pd.read_csv("data/garments_worker_productivity.csv")

print("Dataset shape:", data.shape)


# Create classification target
data["MeetsTarget"] = (
    data["actual_productivity"] >= data["targeted_productivity"]
).astype(int)


# Create features and targets
X = data.drop(columns=["actual_productivity", "MeetsTarget"])

y_regression = data["actual_productivity"].values
y_classification = data["MeetsTarget"].values


# Use the exact same train/test split from Part A
split = np.load("data/split_data.npz")

train_idx = split["train_index"]
test_idx = split["test_index"]

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_reg_train = y_regression[train_idx]
y_reg_test = y_regression[test_idx]

y_class_train = y_classification[train_idx]
y_class_test = y_classification[test_idx]

print("Training rows:", len(train_idx))
print("Testing rows:", len(test_idx))


# Find numerical and categorical columns
numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# Fill missing numerical values using training median
for column in numerical_columns:
    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


# Fill missing categorical values using training mode
for column in categorical_columns:
    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(mode_value)
    X_test[column] = X_test[column].fillna(mode_value)


# Convert categorical columns into dummy variables
X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    drop_first=True
)


# Make sure train and test have the same columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# Convert all columns to numbers
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# Convert to NumPy arrays
X_train = X_train.values
X_test = X_test.values


# Scale features manually
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

train_std[train_std == 0] = 1

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


# Add intercept column
X_train_with_intercept = np.column_stack(
    [np.ones(len(X_train)), X_train]
)

X_test_with_intercept = np.column_stack(
    [np.ones(len(X_test)), X_test]
)


# Linear Regression

start_time = time.perf_counter()

theta = np.linalg.pinv(
    X_train_with_intercept.T @ X_train_with_intercept
) @ X_train_with_intercept.T @ y_reg_train

linear_training_time = time.perf_counter() - start_time


start_time = time.perf_counter()

y_reg_pred = X_test_with_intercept @ theta

linear_prediction_time = time.perf_counter() - start_time


# Calculate regression metrics manually
errors = y_reg_test - y_reg_pred

mae = np.mean(np.abs(errors))

rmse = np.sqrt(np.mean(errors ** 2))

ss_res = np.sum(errors ** 2)

ss_total = np.sum(
    (y_reg_test - np.mean(y_reg_test)) ** 2
)

r2 = 1 - (ss_res / ss_total)


print("\nLinear Regression")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Training time:", linear_training_time)
print("Prediction time:", linear_prediction_time)


# Logistic Regression

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


weights = np.zeros(
    X_train_with_intercept.shape[1]
)

learning_rate = 0.01
iterations = 5000


start_time = time.perf_counter()

for i in range(iterations):

    scores = X_train_with_intercept @ weights

    probabilities = sigmoid(scores)

    gradient = (
        X_train_with_intercept.T
        @ (probabilities - y_class_train)
    ) / len(y_class_train)

    weights = weights - learning_rate * gradient


logistic_training_time = time.perf_counter() - start_time


# Make classification predictions
start_time = time.perf_counter()

test_scores = X_test_with_intercept @ weights

test_probabilities = sigmoid(test_scores)

y_class_pred = (
    test_probabilities >= 0.5
).astype(int)

logistic_prediction_time = time.perf_counter() - start_time


# Calculate classification metrics manually

true_positive = np.sum(
    (y_class_test == 1) &
    (y_class_pred == 1)
)

true_negative = np.sum(
    (y_class_test == 0) &
    (y_class_pred == 0)
)

false_positive = np.sum(
    (y_class_test == 0) &
    (y_class_pred == 1)
)

false_negative = np.sum(
    (y_class_test == 1) &
    (y_class_pred == 0)
)


accuracy = (
    true_positive + true_negative
) / len(y_class_test)


if true_positive + false_positive != 0:
    precision = (
        true_positive /
        (true_positive + false_positive)
    )
else:
    precision = 0


if true_positive + false_negative != 0:
    recall = (
        true_positive /
        (true_positive + false_negative)
    )
else:
    recall = 0


if precision + recall != 0:
    f1 = (
        2 * precision * recall
        / (precision + recall)
    )
else:
    f1 = 0


print("\nLogistic Regression")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Training time:", logistic_training_time)
print("Prediction time:", logistic_prediction_time)


# Final results table

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
        linear_training_time,
        logistic_training_time
    ],

    "Prediction Time": [
        linear_prediction_time,
        logistic_prediction_time
    ]
})


print("\nFinal Part B Results:")
print(results)


# Save CSV file
results.to_csv(
    "data/part_b_results.csv",
    index=False
)


# Save JSON file
part_b_json = {
    "Linear Regression": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Training Time": float(linear_training_time),
        "Prediction Time": float(linear_prediction_time)
    },

    "Logistic Regression": {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Training Time": float(logistic_training_time),
        "Prediction Time": float(logistic_prediction_time)
    }
}


with open(
    "data/part_b_results.json",
    "w"
) as file:
    json.dump(
        part_b_json,
        file,
        indent=4
    )


print("\nPart B completed.")
