# README for Models

This directory contains the models used in the Streamlit application for income prediction based on user input.

## Model Overview

The primary model used in this project is a Logistic Regression model trained on the Adult Income dataset. The model predicts whether an individual's income exceeds $50K based on various demographic and employment features.

## Retraining the Model

To retrain the model, follow these steps:

1. **Update the Dataset**: Ensure that the `data/adult.csv` file is updated with the latest data if necessary.

2. **Run the Training Script**: Execute the `train.py` script to train the model and save the updated model and scaler. This can be done by running the following command in your terminal:

   ```
   python train.py
   ```

3. **Model and Scaler Files**: After training, the model and scaler will be saved in the root directory with a timestamp in their filenames. Ensure that the `model_handler.py` file is updated to load the latest model and scaler.

## Usage

The trained model is loaded and used in the `model_handler.py` file, which provides functions for making predictions based on user input from the Streamlit application.

For any further modifications or updates to the model, refer to the code in `train.py` and `model_handler.py`.