# Streamlit Income Prediction App

This project is a Streamlit application that predicts income categories based on user input. It utilizes a trained machine learning model to provide predictions and is structured to allow for easy updates and retraining of the model.

## Project Structure

```
streamlit-app
├── app.py               # Main entry point for the Streamlit application
├── model_handler.py     # Functions for loading the model and making predictions
├── train.py             # Script for training the model and saving it
├── data
│   └── adult.csv        # Dataset used for training the model
├── models
│   └── README.md        # Documentation for the models used in the project
├── requirements.txt     # List of dependencies for the project
└── README.md            # General documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd streamlit-app
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Streamlit application by running:
   ```
   streamlit run app.py
   ```

## Usage

- Upon running the application, users will be prompted to enter various features related to income prediction.
- After submitting the input, the application will display the predicted income category.

## Retraining the Model

To retrain the model, you can run the `train.py` script. This will load the dataset, train the model, and save the updated model and scaler.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.