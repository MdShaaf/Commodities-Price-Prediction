import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import pickle 
# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('Model Building.log')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'Model Building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Model Building initialized.")
# Example function to demonstrate model building logging
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)
def build_model(data):
    try:
        # Placeholder for model building logic
        logger.info("Starting model building process.")
        print("We have already preprocessed the data for yearly trends and now we will build the model.")
        #Let's build the model
        logger.info("First let's Grid Search for best p,d,q and P,D,Q Values...")
        # Example: Using pmdarima to find the best ARIMA parameters
        best_model = pm.auto_arima(data['Price'], seasonal=True, m=12,
                                   trace=True, error_action='ignore', suppress_warnings=True,stepwise=True,start_P=0, start_Q=0,max_P=2, max_Q=2, max_d=2, max_D=2,
                                   start_p=0, start_q=0, max_p=3, max_q=3)
        logger.info(f"Best model parameters: {best_model.order} Seasonal order: {best_model.seasonal_order}")
        print(f"Best model parameters: {best_model.order} Seasonal order: {best_model.seasonal_order}")
        print("Now let's fit the model and plot the results.")
        arima_model=ARIMA(data['Price'], order=best_model.order)
        arima_result=arima_model.fit()
        plt.figure(figsize=(10,5))
        plt.plot(data.index, data['Price'], label='Original')
        plt.plot(data.index, arima_result.fittedvalues, color='red', label='Fitted')
        plt.title('ARIMA Model Fit')
        plt.legend()
        logger.info("ARIMA model fitted and plotted.")
        #Evaluate the model
        logger.info("Evaluating the model...")
        arima_score=r2_score(data['Price'], arima_result.fittedvalues)
        mse=mean_squared_error(data['Price'], arima_result.fittedvalues)
        mae=mean_absolute_error(data['Price'], arima_result.fittedvalues)
        logger.info(f"MSE of the Arima model: {mse:.2f}")
        logger.info(f"MAE of the Arima model: {mae:.2f}")
        logger.info(f"R2 Score of the Arima model: {arima_score:.2f}")
        
        #let's try SARIMA
        logger.info("Now let's try SARIMA model...")
        sarima_model=SARIMAX(data['Price'], order=best_model.order, seasonal_order=best_model.seasonal_order)
        sarima_result=sarima_model.fit()
        plt.figure(figsize=(10,5))
        plt.plot(data.index, data['Price'], label='Original')
        plt.plot(data.index, sarima_result.fittedvalues, color='green', label='Fitted SARIMA')
        plt.title('SARIMA Model Fit')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, 'SARIMA_Model_Fit.png'))
        logger.info("SARIMA model fitted and plotted.")
                #Evaluate the SARIMA model
        logger.info("Evaluating the SARIMA model...")
        sarima_score=r2_score(data['Price'], sarima_result.fittedvalues)
        sarima_mse=mean_squared_error(data['Price'], sarima_result.fittedvalues)
        sarima_mae=mean_absolute_error(data['Price'], sarima_result.fittedvalues)
        logger.info(f"SARIMA MSE of the Sarimax model: {sarima_mse:.2f}")
        logger.info(f"SARIMA MAE of the Sarimax model: {sarima_mae:.2f}")
        logger.info(f"SARIMA R2 Score of the Sarimax model: {sarima_score:.2f}")

        #choose the best model based on R2 Score
        # Choose and save the best model based on R2 Score
        if sarima_score > arima_score:
            logger.info("SARIMAX model performs better than ARIMA model.")
            print("SARIMAX model performs better than ARIMA model.")
            best_result = sarima_result
            model_name = 'SARIMAX_Model.pkl'
        else:
            logger.info("ARIMA model performs better than SARIMAX model.")
            print("ARIMA model performs better than SARIMAX model.")
            best_result = arima_result
            model_name = 'ARIMA_Model.pkl'

        # Save the model
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(best_result, f)

        logger.info(f"Model saved at {model_path}")
        print(f"Model saved at {model_path}")

        return best_result  # Now return after saving
        
        # Simulate model building steps
        # e.g., splitting data, training model, evaluating model
        logger.info("Model built successfully.")
    except Exception as e:
        logger.error(f"Error during model building: {e}")
        raise
       
data=pd.read_csv(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Preprocessed\preprocessed_data.csv",parse_dates=['Date'], index_col='Date')
build_model(data)
