import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('Data_Preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'Data_Preprocessing')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Data Preprocessing module initialized.")

#Reading the data
data_file_path = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Raw\agmarknet_data.xlsx"
try:
    """Loads the data from Excel file."""
    df = pd.read_excel(data_file_path)
    logger.info(f"Data loaded successfully from {data_file_path}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise
# Display basic information about the dataset
logger.info(f"Dataset shape: {df.shape}")

#Checking for missing values
missing_values = df.isnull().sum()

# Handling missing values
# Dropping rows with missing values for simplicity
logger.info("Checking for missing values in each column:")
if missing_values.any():
    logger.info("Missing values found:Removing rows with missing values.")
    df.dropna(inplace=True)
    logger.info("Dropped rows with missing values.")
    current_missing_values = df.isnull().sum()
    logger.info(f"Current missing values after dropping: {current_missing_values}")
else:
    logger.info("No missing values found.")

# Checking for duplicate rows
duplicate_rows = df.duplicated().sum()
if duplicate_rows > 0: # If there are duplicate rows
    logger.info(f"Found {duplicate_rows} duplicate rows. Removing duplicates.")
    df.drop_duplicates(inplace=True)
    logger.info("Duplicate rows removed.")
else:
    logger.info("No duplicate rows found.")

df.rename(columns={0:'Sl.No',1:'District Name',2:'Market Name',3:'Commodity',4:'Variety',5:'Grade',6:'Min Price',7:'Max Price',8:'Avg Price',9:'Date'}, inplace=True) 
logger.info(f"Found columns : \n{df.columns}")   
# Display basic statistics of the dataset
# logger.info(f"Dataset statistics:\n{df.describe()}")
print("Checking the different Grades of commodities available in the dataset")
print(df['Grade'].value_counts())
print("Selecting only 'FAQ' variety for the analysis")
df = df[df['Grade'] == 'FAQ']

print("As we have 3 different prices,let's plot and see the differences.")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
min_resampled=df['Min Price'].resample('ME').mean() # Minimum price Monthly resampling
max_resampled=df['Max Price'].resample('ME').mean() # Maximum price Monthly resampling
avg_resampled=df['Avg Price'].resample('ME').mean() # Average price Monthly resampling

#let's set up plot directory
plot_dir = 'plots'
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(15,6))
plt.plot(min_resampled, label='Min Price', color='blue')
plt.plot(max_resampled, label='Max Price', color='red')
plt.plot(avg_resampled, label='Avg Price', color='green')
plt.title('Monthly Resampled Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'weekly_prices.png'))
logger.info("Plotted Min, Max and Avg Prices for comparison.")
plt.close()

print("Selecting Average Price for the analysis")
df = df[['Avg Price']]
df.rename(columns={'Avg Price':'Price'}, inplace=True)
logger.info(f"Data after selecting 'FAQ' grade and 'Avg Price':\n{df.head()}")
df.sort_index(inplace=True)
# Checking for outliers using boxplot
df=df.resample('ME').mean() # Monthly resampling
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Price'])
plt.title('Boxplot of Prices')
plt.savefig(os.path.join(plot_dir, 'Boxplot_of_Prices.png'))
logger.info("Plotted Boxplot of Prices to check for outliers.")
plt.figure(figsize=(15,6))
plt.plot(df['Price'], label='Price', color='blue')
plt.title('01-Monthly Resampled Average Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(plot_dir, '01-Monthly Resampled Average Prices.png'))
logger.info("Plotted Monthly Resampled Average Prices.")

#There are not entries between 2012 to 2015
#So we will consider data from 2016 to till date
df=df[df.index >= '2016-01-01']
# df=df.to_frame()
#for random missing values we can use interpolation
df['Price'].interpolate(method='linear', inplace=True)
plt.plot(figsize=(15,6))
plt.plot(df['Price'], label='Price', color='blue')
plt.title('02-Monthly Resampled Average Prices from 2016 onwards')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(plot_dir, '02-Monthly Resampled Average Prices from 2016 onwards.png'))
logger.info("Plotted Monthly Resampled Average Prices from 2016 onwards.")
logger.info(f"Data after filtering from 2016 onwards:\n{df.head()}")
# Checking for stationarity using ADF test

result = adfuller(df['Price'])
logger.info('ADF Statistic: %f' % result[0])
logger.info('p-value: %f' % result[1])
pvalue=result[1]
if pvalue <= 0.05:
    logger.info("The time series is stationary.")
else:
    logger.info("The time series is non-stationary, applying differencing.")

#checking for seasonality and trend using seasonal decomposition
decomposition = seasonal_decompose(df['Price'], model='additive', period=52)
decomposition.plot()
plt.suptitle('Seasonal Decomposition of Prices', fontsize=16)
plt.savefig(os.path.join(plot_dir, '03-Seasonal Decomposition of Prices.png'))
logger.info("Plotted Seasonal Decomposition of Prices.")

#applying log transformation
df['Price'] = np.log(df['Price'])
plt.figure(figsize=(15,6))
plt.plot(df)
plt.title(' 03-Monthly Log Transformed Resampled Average Prices from 2016 onwards')
plt.xlabel('Date')
plt.ylabel('Log Price')
plt.legend()    
plt.savefig(os.path.join(plot_dir, ' 03-Monthly Log Transformed Resampled Average Prices from 2016 onwards.png'))
logger.info("Plotted Monthly Log Transformed Resampled Average Prices from 2016 onwards.")
df.to_csv(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Preprocessed\preprocessed_data.csv")
logger.info("Preprocessed data saved to 'preprocessed_data.csv'.")