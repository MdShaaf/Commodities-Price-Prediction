import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import warnings
import pickle
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('Forecasting.log')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'Forecasting.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Beginning the Forecast.")
print("Beginning the Forecast. We will use the saved model to forecast future prices.")

def load_model(model_dir='models'):
    """Load the saved model from pickle file."""
    try:
        # Check which model exists
        sarima_path = os.path.join(model_dir, 'SARIMAX_Model.pkl')
        arima_path = os.path.join(model_dir, 'ARIMA_Model.pkl')
        
        if os.path.exists(sarima_path):
            model_path = sarima_path
            logger.info("Loading SARIMAX model...")
        elif os.path.exists(arima_path):
            model_path = arima_path
            logger.info("Loading ARIMA model...")
        else:
            raise FileNotFoundError("No trained model found. Please run Model_Building.py first.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {model_path}")
        print(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def forecast_prices(data, steps=30):
    try:
        logger.info("Starting the forecasting process.")
        print("Using the saved model to forecast future prices.")
        
        # Load the saved model instead of building it
        model = load_model()
        
        # Forecast future prices
        forecast = model.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Generate future dates
        future_dates = pd.date_range(start=data.index[-1], periods=steps + 1, freq='W')[1:]
        
        # Combine into DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_mean,
            'Lower CI': conf_int.iloc[:, 0],
            'Upper CI': conf_int.iloc[:, 1]
        })
        forecast_df.set_index('Date', inplace=True)
        
        logger.info("Forecasting completed successfully.")
        print("Forecasting completed successfully. Here are the forecasted prices:")
        print(forecast_df.head())
        
        # Create interactive Plotly figure
        fig = go.Figure()
        
        # Converting data to the normal scale from the log scale
        data_actual_forecast = pd.DataFrame(np.exp(forecast_df['Forecast']))
        data_exp = pd.DataFrame(np.exp(data['Price']))
        
        # Historical data (last 50 points)
        data_exp = data_exp.tail(50)
        fig.add_trace(go.Scatter(
            x=data_exp.index,
            y=data_exp['Price'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black')
        ))

        # Forecasted data
        fig.add_trace(go.Scatter(
            x=data_actual_forecast.index,
            y=data_actual_forecast['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue')
        ))

        # Layout settings
        fig.update_layout(
            title="Interactive Time Series Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        fig.show()
        
        return forecast_df

    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        print(f"An error occurred during forecasting: {e}")

if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Preprocessed\preprocessed_data.csv", 
                       parse_dates=['Date'], index_col='Date')
    forecast_prices(data, steps=30)