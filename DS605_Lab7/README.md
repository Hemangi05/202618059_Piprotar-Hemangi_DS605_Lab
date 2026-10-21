# DS605 Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

## Overview

This lab explores image representation and face recognition using the ORL face image dataset. It covers image preprocessing, Principal Component Analysis (PCA), eigenfaces, image reconstruction, dimensionality reduction, and traditional machine learning classifiers.

The main objective is to understand how face images can be represented using fewer features while retaining useful information for recognition.

## Dataset

The project uses the ORL face image dataset, which contains grayscale face images belonging to different subjects.
Dataset link : https://www.kaggle.com/datasets/kasikrit/att-database-of-faces

The standard dataset contains:
- 40 subjects
- 10 images per subject
- 400 images in total
- Image dimensions of 112 × 92 pixels

Each image is converted into a one-dimensional pixel vector for numerical processing. Subject identities are retained as class labels.

*Note: The dataset statistics above describe the standard ORL dataset. Actual results depend on the extracted dataset used in the project.*

## Objectives

- Load and inspect grayscale face images.
- Convert images into numerical feature vectors.
- Divide the dataset into training and testing sets while representing every subject in both sets.
- Implement PCA using NumPy linear algebra.
- Compare covariance eigendecomposition and Singular Value Decomposition (SVD).
- Calculate cumulative explained variance and select enough components to preserve at least 90% of the variance.
- Visualize eigenfaces and reconstruct sample images.
- Compare the storage requirements of original images and PCA representations.
- Perform nearest-face recognition using Euclidean distance.
- Train and evaluate traditional machine learning classifiers using PCA features.

## Implementation

### Part A: Image Representation

The images are loaded as grayscale arrays and flattened into one-dimensional feature vectors. These vectors are combined into a feature matrix, with subject identities stored separately as labels.

A reproducible, subject-wise train/test split is used so that each subject appears in both sets.

### Part B: PCA, Eigenfaces and Compression

PCA is implemented using NumPy rather than a ready-made PCA estimator.

The steps include:
1. Calculate the mean face from the training images.
2. Center the training data.
3. Compute principal directions using covariance eigendecomposition and SVD.
4. Compare the directions obtained from both methods.
5. Plot cumulative explained variance.
6. Select the minimum number of components needed to preserve at least 90% variance.
7. Visualize eigenfaces and project images into PCA space.
8. Reconstruct sample faces and measure reconstruction error using mean squared error.
9. Compare the storage requirements of original pixel data and PCA representations.

### Part C: Recognition in PCA Space

Euclidean nearest-neighbour matching is used as a baseline face recognition method.

Recognition is evaluated using two representations:
- Original pixel features.
- Reduced PCA features.

The recognition accuracies are compared, and selected correct and incorrect PCA matches are visualized.

### Part D: Traditional Machine Learning

Three traditional classifiers are evaluated using PCA features:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)

Different numbers of principal components are tested to examine the trade-off between recognition accuracy and representation size.

The classifier results are compared with the nearest-face recognition baseline.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- Jupyter Notebook
- Visual Studio Code


## Conclusion

This lab demonstrates how PCA can transform high-dimensional face images into a lower-dimensional representation. Eigenfaces provide a way to visualize the principal patterns of variation, while image reconstruction illustrates the information retained and lost during dimensionality reduction.

Nearest-neighbour matching and traditional machine learning classifiers provide different approaches to face recognition. Comparing their results helps evaluate the relationship between feature representation, recognition accuracy, reconstruction quality, and storage requirements.

## Academic Context

**Course:** DS605 — Fundamentals of Machine Learning  
**Lab:** Lab 7 — Image Representation, PCA, Eigenfaces and Face Recognition