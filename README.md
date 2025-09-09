# Stock Price Prediction with LSTM

##  Overview

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next-day closing price of a stock. It fetches historical data from Yahoo Finance, engineers several technical indicator features, and trains a Keras/TensorFlow model. The project is deployed as a simple, interactive web application using Streamlit.

---

## Features

  **Data Collection**: Fetches daily stock data (Open, High, Low, Close) using the yfinance library.
  **Feature Engineering**: Creates technical indicators like Moving Averages (MA20, MA50) and the Relative Strength Index (RSI) to provide more context to the model.
  **LSTM Model**: Builds a sequential model with two LSTM layers and Dropout for regularization.
  **Evaluation**: Assesses model performance using Root Mean Squared Error (RMSE) and visualizes the predicted vs. actual prices.
  **Interactive App**: A Streamlit application (app.py) allows users to input any stock ticker and get a live prediction.

---

## How to Run

1.  **Clone the repository:**
   
bash
    git clone <your-repo-link>
    cd <your-repo-folder>
   

2.  **Create and activate a virtual environment:**
   
bash
    # Using Conda
    conda create --name stock_env python=3.10
    conda activate stock_env
   

3.  **Install dependencies:**
   
bash
    pip install -r requirements.txt
   

4.  **(Optional) Retrain the model:**
    Open and run the Stock_Price_Prediction.ipynb notebook to see the full training and evaluation process. This will overwrite the saved model file.

5.  **Run the Streamlit app:**
   
bash
    streamlit run app.py
   
    Open the URL provided by Streamlit in your browser to interact with the application.

---

## File Structure

├── Stock_Price_Prediction.ipynb  # Notebook with the full code and analysis
├── app.py                        # The Streamlit application script
├── lstm_full_model.keras         # The trained and saved Keras model
├── predictions.csv               # Sample predictions on the test set
├── requirements.txt              # Required Python packages
├── report.pdf                    # Project summary report
└── README.md                     # This file
