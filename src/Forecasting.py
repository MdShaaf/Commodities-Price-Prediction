import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from Model_Building import build_model
warnings.filterwarnings("ignore")
import plotly.graph_objects as go


# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

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
print("Beginning the Forecast. We will use the model we built to forecast future prices.")
def forecast_prices(data, steps=30):
    try:
        logger.info("Starting the forecasting process.")
        print("Using the ARIMA model to forecast future prices.")
        # Build the model using the function from Model_Building
        model = build_model(data)
        
        # Forecast future prices
        forecast = model.get_forecast(steps=steps,frequency='W')
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
        print(forecast_df)
        
        #-------------------------------
        # Create interactive Plotly figure
        # -------------------------------
        fig = go.Figure()
        #converting data to the normal scale from the log scale
        data_actual_forcast=pd.DataFrame(np.exp(forecast_df['Forecast']))
        data=pd.DataFrame(np.exp(data['Price']))
        # Historical data
        data=data.tail(50)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Price'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black')
        ))

        # Forecasted data
        fig.add_trace(go.Scatter(
            x=data_actual_forcast.index,
            y=data_actual_forcast['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue')
        ))

        # Layout settings
        fig.update_layout(
            title="Interactive Time Series Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified")

# -------------------------------
# Show interactive figure
# -------------------------------
        fig.show()

    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        print(f"An error occurred during forecasting: {e}")

data = pd.read_csv(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\preprocessed_data.csv", parse_dates=['Date'], index_col='Date')
forecast_prices(data, steps=30)