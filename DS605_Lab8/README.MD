# DS605 Lab Assignment 8: Decision Trees and Support Vector Machines

## Overview

This assignment explores two important supervised machine learning algorithms: **Decision Trees** and **Support Vector Machines (SVMs)**. It focuses on understanding the underlying mathematical concepts, implementing algorithms from scratch, visualizing decision boundaries, experimenting with hyperparameters, and comparing custom implementations with Scikit-learn.

The assignment is divided into four parts, covering decision tree learning, linear SVM optimization, kernel-based classification, and comprehensive model comparison.

## Objectives

- Understand impurity measures and information gain used in Decision Trees.
- Implement a Decision Tree classifier from scratch.
- Understand hinge loss, regularization, and margin-based classification in SVMs.
- Implement a Linear SVM using subgradient descent.
- Compare different SVM kernels for nonlinear classification.
- Visualize decision boundaries and evaluate classification models.
- Investigate the effects of hyperparameters on model performance.
- Compare models based on predictive performance and computational efficiency.

## Part A — Decision Trees

This part focuses on understanding and implementing Decision Tree classification using the Banknote Authentication dataset.

### Topics Covered

- Dataset loading and preprocessing.
- Selection of features and splitting data into training and testing sets.
- Visualization of the dataset using scatter plots.
- Implementation of Gini impurity and entropy.
- Calculation of information gain to select the best split.
- Construction of a recursive Decision Tree classifier from scratch.
- Prediction and evaluation of the manually implemented tree.
- Visualization of tree structure and decision boundaries.
- Comparison with Scikit-learn's `DecisionTreeClassifier`.

### Experiments

- Investigate the effect of maximum tree depth.
- Compare Gini impurity and entropy as splitting criteria.
- Compare the manual implementation with Scikit-learn in terms of performance and execution time.

## Part B — Linear Support Vector Machine

This part explores Linear SVM classification, focusing on hinge loss, margin maximization, and optimization using subgradient descent.

### Topics Covered

- Feature standardization using training-set statistics.
- Conversion of binary class labels to \(-1\) and \(+1\).
- Implementation of hinge loss and the SVM objective function.
- Development of a manual Linear SVM classifier.
- Training using subgradient descent.
- Visualization of the separating hyperplane, margins, and margin-violating samples.
- Comparison with Scikit-learn's `SVC` using a linear kernel.

### Experiments

- Investigate the effect of regularization parameter \(C\).
- Study the influence of learning rate on optimization.
- Examine how the number of iterations affects convergence.
- Compare classification performance and execution time.

## Part C — SVM Kernel Comparison

This part investigates how kernel functions enable SVMs to classify data with nonlinear decision boundaries.

A synthetic nonlinear dataset is generated using Scikit-learn's `make_moons` function.

### Models Compared

- **Linear Kernel:** Creates a linear decision boundary.
- **Polynomial Kernel:** Models nonlinear relationships using polynomial transformations.
- **RBF Kernel:** Captures nonlinear patterns using a radial basis function.

### Topics Covered

- Generation and visualization of synthetic data.
- Feature standardization without test-set data leakage.
- Training SVM classifiers with different kernels.
- Visualization and comparison of decision boundaries.
- Evaluation of classification performance.

### Experiments

- Compare the performance of Linear, Polynomial, and RBF kernels.
- Investigate the effects of regularization parameter \(C\) and RBF kernel parameter gamma.
- Examine how kernel selection influences the decision boundary and classification results.

## Part D — Comprehensive Model Comparison

This part compares five classification models using **all four features** of the Banknote Authentication dataset.

### Models Compared

1. Manual Decision Tree.
2. Scikit-learn Decision Tree.
3. Manual Linear SVM.
4. Scikit-learn Linear SVM.
5. Scikit-learn RBF SVM.

### Evaluation

Each model is evaluated using the same training and testing split to support a fair comparison.

The following metrics are reported:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Training time:** Time required to fit the model.
- **Prediction time:** Time required to generate test-set predictions.

The results are organized into a comparison table to help identify differences in predictive performance and computational cost.

## Datasets

### 1. Banknote Authentication Dataset

Used in Parts A, B, and D.

The dataset contains four numerical features extracted from images of genuine and forged banknotes:

- `variance`
- `skewness`
- `curtosis`
- `entropy`

The target variable, `class`, represents the binary banknote classification label.

### 2. Synthetic Nonlinear Dataset

Used in Part C.

A two-dimensional dataset is generated using Scikit-learn's `make_moons` function to investigate the behavior of different SVM kernels on nonlinear classification problems.

## Technologies and Libraries

- **Python** — Implementation of machine learning algorithms.
- **NumPy** — Numerical computation and array operations.
- **Pandas** — Dataset loading and result analysis.
- **Matplotlib** — Data visualization and decision boundary plots.
- **Scikit-learn** — Reference implementations, preprocessing, dataset generation, and evaluation metrics.
- **Jupyter Notebook** — Interactive experimentation and documentation.



## Key Learning Outcomes

By completing this assignment, the following concepts are explored:

- Building a Decision Tree using impurity reduction and information gain.
- Understanding the mathematical foundations of Linear SVM classification.
- Implementing an optimization procedure using subgradient descent.
- Understanding the role of kernel functions in nonlinear classification.
- Visualizing decision boundaries to interpret classifier behavior.
- Evaluating models using multiple classification metrics.
- Comparing custom implementations with established machine learning libraries.
- Understanding the trade-offs between model performance, complexity, and computational cost.

## Conclusion

This assignment provides practical experience with Decision Trees and Support Vector Machines, from implementing core algorithms to evaluating and comparing trained models. It emphasizes both the mathematical foundations of classification and the practical considerations involved in selecting an appropriate machine learning model.

