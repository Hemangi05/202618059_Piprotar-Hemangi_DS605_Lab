import json
import time
import numpy as np
import pandas as pd


# Load Part A and Part B results
with open("data/part_a_results.json", "r") as file:
    part_a = json.load(file)

with open("data/part_b_results.json", "r") as file:
    part_b = json.load(file)


print("Part A Results:")
print(json.dumps(part_a, indent=4))

print("\nPart B Results:")
print(json.dumps(part_b, indent=4))


# Find Logistic Regression result from Part A
part_a_logistic = None

for result in part_a:
    if result["Model"] == "Logistic Regression":
        part_a_logistic = result
        break


# Load dataset
data = pd.read_csv("data/garments_worker_productivity.csv")

print("\nDataset shape:", data.shape)


# Create classification target
data["MeetsTarget"] = (
    data["actual_productivity"] >= data["targeted_productivity"]
).astype(int)


# Load the same train-test split used in Part A
split = np.load("data/split_data.npz")

train_index = split["train_index"]
test_index = split["test_index"]


train_data = data.iloc[train_index].copy()
test_data = data.iloc[test_index].copy()

print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))


# Separate target
y_train = train_data["MeetsTarget"].values
y_test = test_data["MeetsTarget"].values


# Do not use actual_productivity as a classification feature
drop_columns = ["actual_productivity", "MeetsTarget"]

X_train = train_data.drop(columns=drop_columns)
X_test = test_data.drop(columns=drop_columns)


# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()


print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# Fill numerical missing values using training medians
for column in numerical_columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


# Fill categorical missing values using training modes
for column in categorical_columns:

    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(mode_value)
    X_test[column] = X_test[column].fillna(mode_value)


# Convert categorical variables into dummy variables
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


# Make sure test data has exactly the same columns as training data
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# Convert boolean columns to numbers
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# Standardization using training data only
mean = X_train.mean()
std = X_train.std()

std = std.replace(0, 1)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# Convert to NumPy arrays
X_train = X_train.values
X_test = X_test.values


# Add intercept
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# Sigmoid function
def sigmoid(z):

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# Optimized Logistic Regression
start_time = time.perf_counter()

weights = np.zeros(X_train.shape[1])

learning_rate = 0.05
iterations = 3000

for i in range(iterations):

    probabilities = sigmoid(X_train @ weights)

    error = probabilities - y_train

    gradient = (X_train.T @ error) / len(y_train)

    weights -= learning_rate * gradient


training_time = time.perf_counter() - start_time


# Prediction
start_time = time.perf_counter()

test_probabilities = sigmoid(X_test @ weights)

y_pred = (test_probabilities >= 0.5).astype(int)

prediction_time = time.perf_counter() - start_time


# Manual classification metrics
true_positive = np.sum((y_test == 1) & (y_pred == 1))
true_negative = np.sum((y_test == 0) & (y_pred == 0))
false_positive = np.sum((y_test == 0) & (y_pred == 1))
false_negative = np.sum((y_test == 1) & (y_pred == 0))


accuracy = (true_positive + true_negative) / len(y_test)


if true_positive + false_positive > 0:
    precision = true_positive / (true_positive + false_positive)
else:
    precision = 0


if true_positive + false_negative > 0:
    recall = true_positive / (true_positive + false_negative)
else:
    recall = 0


if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0


# Display optimized results
print("\nOptimized Manual Logistic Regression")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Training time:", training_time)
print("Prediction time:", prediction_time)


# Find Logistic Regression result from Part A
part_a_logistic = None

for result in part_a:
    if result["Model"] == "Logistic Regression":
        part_a_logistic = result
        break


# Create comparison
comparison = {
    "Sklearn Logistic Regression": {
        "Accuracy": part_a_logistic["Accuracy"],
        "Precision": part_a_logistic["Precision"],
        "Recall": part_a_logistic["Recall"],
        "F1": part_a_logistic["F1"],
        "Training Time": part_a_logistic["Training Time"],
        "Prediction Time": part_a_logistic["Prediction Time"]
    },

    "Manual Logistic Regression": {
        "Accuracy": part_b["Logistic Regression"]["Accuracy"],
        "Precision": part_b["Logistic Regression"]["Precision"],
        "Recall": part_b["Logistic Regression"]["Recall"],
        "F1": part_b["Logistic Regression"]["F1"],
        "Training Time": part_b["Logistic Regression"]["Training Time"],
        "Prediction Time": part_b["Logistic Regression"]["Prediction Time"]
    },

    "Optimized Manual Logistic Regression": {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Training Time": training_time,
        "Prediction Time": prediction_time
    }
}


print("\nComparison:")
print(json.dumps(comparison, indent=4))


# Save comparison
with open("data/part_c_comparison.json", "w") as file:
    json.dump(comparison, file, indent=4)


print("\nPart C completed.")