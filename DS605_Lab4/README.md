# Airbnb Price Prediction

## Project Overview

This project develops a machine learning model to predict the nightly price of Airbnb listings using the AB_NYC_2019 dataset.

## Project Workflow

The project includes data cleaning, exploratory data analysis, feature engineering, preprocessing, regression model comparison, hyperparameter tuning, and model evaluation.

## Models Used

The following regression models were compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Random Forest was further optimized using RandomizedSearchCV.

## Feature Engineering

The project uses features such as:

- Neighbourhood
- Neighbourhood group
- Room type
- Latitude and longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Availability
- Review recency
- Review availability ratio
- Log-transformed minimum nights

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The final tuned Random Forest model achieved:

- MAE: 31.37
- RMSE: 43.49
- R² Score: 0.58

## Streamlit Application

A Streamlit web application was developed to allow users to enter Airbnb listing information and receive an estimated nightly price.

## Files

- `DS605_202618059_....ipynb` - Complete machine learning workflow
- `app.py` - Streamlit application
- `airbnb_price_model.pkl` - Saved trained model
- `model_results.csv` - Model comparison results
- `AB_NYC_2019.csv` - Dataset
- `requirements.txt` - Required Python packages

## Limitations

The model is trained using historical Airbnb data from New York City. Therefore, predictions may not accurately represent current prices, prices in other cities, or future market conditions.

Other factors such as seasonality, demand, special events, and changes in the Airbnb market may also affect actual prices.
