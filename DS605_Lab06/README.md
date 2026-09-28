# DS605 Lab 6 - Feature Extraction and Machine Learning

## Dataset

Two datasets were used:

- **Asphalt Crack Dataset** - 400 images
  - 200 Crack images
  - 200 Non-crack images
- **Email Spam Dataset** - 5,172 emails
  - 3,672 Non-spam
  - 1,500 Spam

## Part A - Image Classification

The asphalt images were resized to 224 × 224 and converted to grayscale.

The following features were extracted:

- Mean brightness
- Contrast
- Minimum and maximum intensity
- Median intensity
- Dark pixel ratio
- Bright pixel ratio
- Canny edge count
- Canny edge density

Two models were tested:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 93.75% | 93.83% |
| Random Forest | 95.00% | 95.00% |

Random Forest gave slightly better classification results, but it took more time to train and predict than Logistic Regression.

## Part B - Email Classification

The provided email dataset already contained 3,000 word-count features. These were used as the count-based representation.

TF-IDF was also applied to the same features using `TfidfTransformer`.

| Representation | Accuracy | F1 Score | Training Time |
|---|---:|---:|---:|
| Count Features | 98.26% | 97.04% | 7.28 s |
| TF-IDF | 95.07% | 91.34% | 0.10 s |

The count representation gave better classification results in this experiment. TF-IDF was much faster to train.

## Part C - Feature Improvement

The Canny thresholds were changed from `(100, 200)` to `(50, 150)`.

The number of features remained 9.

| Representation | Accuracy | F1 Score |
|---|---:|---:|
| Original Canny | 95.00% | 95.00% |
| Improved Canny | 95.00% | 95.00% |

The change did not improve accuracy, but it showed how changing the feature representation can affect the extracted image information without increasing the number of features.

## Observations

- Simple image features were able to classify crack and non-crack images reasonably well.
- Random Forest performed slightly better than Logistic Regression on the image features.
- The count-based email representation performed better than TF-IDF in this experiment.
- TF-IDF had much lower training and prediction time.
- Changing the Canny thresholds did not change the classification performance.
- Feature representation can affect both model performance and computation time.

