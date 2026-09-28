import pandas as pd
import numpy as np
import time
import json


# Manual implementation uses only NumPy and Pandas.
# The same split, features, targets, and metrics are used for a fair comparison.


# Load raw data
data = pd.read_csv(
    "data/garments_worker_productivity.csv"
)

print("Dataset shape:", data.shape)


# Create classification target
data["MeetsTarget"] = (
    data["actual_productivity"]
    >= data["targeted_productivity"]
).astype(int)


# Separate features and targets
X = data.drop(
    columns=[
        "actual_productivity",
        "MeetsTarget"
    ]
)

y_regression = data["actual_productivity"].values
y_classification = data["MeetsTarget"].values


# Use the exact same split created in Part A
split = np.load(
    "data/split_data.npz"
)

train_index = split["train_index"]
test_index = split["test_index"]


X_train = X.iloc[train_index].copy()
X_test = X.iloc[test_index].copy()

y_reg_train = y_regression[train_index]
y_reg_test = y_regression[test_index]

y_class_train = y_classification[train_index]
y_class_test = y_classification[test_index]


print("Training rows:", len(train_index))
print("Testing rows:", len(test_index))


# Identify column types
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


# Handle missing numerical values
for column in numerical_columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(
        median_value
    )

    X_test[column] = X_test[column].fillna(
        median_value
    )


# Handle missing categorical values
for column in categorical_columns:

    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(
        mode_value
    )

    X_test[column] = X_test[column].fillna(
        mode_value
    )


# Convert categorical variables to dummy variables
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


# Convert values to float
X_train = X_train.astype(float).values
X_test = X_test.astype(float).values


# Scale features manually
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

train_std[train_std == 0] = 1

X_train = (
    X_train - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std


# Add intercept
X_train_with_intercept = np.column_stack(
    [
        np.ones(len(X_train)),
        X_train
    ]
)

X_test_with_intercept = np.column_stack(
    [
        np.ones(len(X_test)),
        X_test
    ]
)


# Linear Regression
start_time = time.perf_counter()

theta = np.linalg.pinv(
    X_train_with_intercept.T
    @ X_train_with_intercept
) @ X_train_with_intercept.T @ y_reg_train

linear_training_time = (
    time.perf_counter()
    - start_time
)


# Linear Regression prediction
start_time = time.perf_counter()

y_reg_pred = (
    X_test_with_intercept @ theta
)

linear_prediction_time = (
    time.perf_counter()
    - start_time
)


# Linear Regression metrics
errors = (
    y_reg_test - y_reg_pred
)

linear_mae = np.mean(
    np.abs(errors)
)

linear_rmse = np.sqrt(
    np.mean(errors ** 2)
)

ss_res = np.sum(
    errors ** 2
)

ss_total = np.sum(
    (
        y_reg_test
        - np.mean(y_reg_test)
    ) ** 2
)

linear_r2 = (
    1 - ss_res / ss_total
)


print("\nLinear Regression")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2:", linear_r2)
print("Training time:", linear_training_time)
print("Prediction time:", linear_prediction_time)


# Sigmoid function
def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500
    )

    return 1 / (
        1 + np.exp(-z)
    )


# Logistic Regression
weights = np.zeros(
    X_train_with_intercept.shape[1]
)

learning_rate = 0.01
iterations = 5000


start_time = time.perf_counter()

for i in range(iterations):

    scores = (
        X_train_with_intercept
        @ weights
    )

    probabilities = sigmoid(
        scores
    )

    gradient = (
        X_train_with_intercept.T
        @ (
            probabilities
            - y_class_train
        )
    ) / len(y_class_train)

    weights = (
        weights
        - learning_rate * gradient
    )


logistic_training_time = (
    time.perf_counter()
    - start_time
)


# Logistic Regression prediction
start_time = time.perf_counter()

test_scores = (
    X_test_with_intercept
    @ weights
)

test_probabilities = sigmoid(
    test_scores
)

y_class_pred = (
    test_probabilities >= 0.5
).astype(int)

logistic_prediction_time = (
    time.perf_counter()
    - start_time
)


# Classification metrics
true_positive = np.sum(
    (y_class_test == 1)
    & (y_class_pred == 1)
)

true_negative = np.sum(
    (y_class_test == 0)
    & (y_class_pred == 0)
)

false_positive = np.sum(
    (y_class_test == 0)
    & (y_class_pred == 1)
)

false_negative = np.sum(
    (y_class_test == 1)
    & (y_class_pred == 0)
)


logistic_accuracy = (
    true_positive + true_negative
) / len(y_class_test)


if true_positive + false_positive != 0:

    logistic_precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
    )

else:

    logistic_precision = 0


if true_positive + false_negative != 0:

    logistic_recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
    )

else:

    logistic_recall = 0


if (
    logistic_precision
    + logistic_recall
) != 0:

    logistic_f1 = (
        2
        * logistic_precision
        * logistic_recall
        / (
            logistic_precision
            + logistic_recall
        )
    )

else:

    logistic_f1 = 0


print("\nLogistic Regression")
print("Accuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1-score:", logistic_f1)
print("Training time:", logistic_training_time)
print("Prediction time:", logistic_prediction_time)


# Final results
results = {
    "Linear Regression": {
        "MAE": float(linear_mae),
        "RMSE": float(linear_rmse),
        "R2": float(linear_r2),
        "Training Time": float(
            linear_training_time
        ),
        "Prediction Time": float(
            linear_prediction_time
        )
    },

    "Logistic Regression": {
        "Accuracy": float(
            logistic_accuracy
        ),
        "Precision": float(
            logistic_precision
        ),
        "Recall": float(
            logistic_recall
        ),
        "F1": float(
            logistic_f1
        ),
        "Training Time": float(
            logistic_training_time
        ),
        "Prediction Time": float(
            logistic_prediction_time
        )
    }
}


print("\nFinal Part B Results:")
print(
    json.dumps(
        results,
        indent=4
    )
)


# Save JSON
with open(
    "data/part_b_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\nPart B completed.")