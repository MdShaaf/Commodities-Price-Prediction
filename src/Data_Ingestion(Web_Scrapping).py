from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select, WebDriverWait
import time
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os
import logging
from tqdm import tqdm

# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('DataIngestionLogger')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'DataIngestionLogger.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#checking if the data file already exists
file_name = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Raw\agmarknet_data new.xlsx"
if os.path.exists(file_name):
    print(f"Data File  already exists. Exiting to avoid overwriting.")
    # exit()
else:
    print(f"File '{file_name}' does not exist. Proceeding with web scraping.")   
    #IF NOT, continue with the web scraping process
    # Create a fresh Chrome session
    # Configure Chrome for headless mode
    logger.info("Configuring Chrome in headless mode.")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # or "--headless=new" if it works in your setup
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])  # hides DevTools logs

    # Start Chrome session
    logger.info("Starting a new headless Chrome session.")
    service = Service()  # specify chromedriver path if needed
    driver = webdriver.Chrome(service=service, options=chrome_options)
    logger.info("Chrome session started successfully.")

    # Open website
    logger.info("Navigating to agmarknet.gov.in")
    driver.get("https://agmarknet.gov.in/")
    logger.info("Navigation successful.")

    logger.info("Navigated to agmarknet.gov.in")
    # Wait for page load
    time.sleep(3)
    # Interacting with the dropdown buttons
    logger.info("Interacting with dropdown menus, Selecting Commodity(Ginger)")
    select_element = driver.find_element(By.ID, "ddlCommodity") #Commodity section
    select = Select(select_element)
    select.select_by_value("103") #selecting the commodity(Ginger)
    time.sleep(2)

    logger.info("Selecting States")
    # Interact with another dropdown,slecting state, district and market
    select_element = driver.find_element(By.ID, "ddlState")
    select = Select(select_element)
    select.select_by_value("KK") #selecting the state(Karnataka)
    time.sleep(3)

    logger.info("Selecting District(Bangalore) and Market(Bangalore)")
    select_element = driver.find_element(By.ID, "ddlDistrict") #District section
    select = Select(select_element)
    select.select_by_value("1") #selecting the district(Bagalore)
    time.sleep(3)

    select_element = driver.find_element(By.ID, "ddlMarket") #Market section
    select = Select(select_element)
    select.select_by_value("107") #selecting the market(Bangalore)
    time.sleep(3)

    #enter date in DD-MM-YYYY format
    logger.info("Entering Date(01-01-2010)")
    date_input = driver.find_element(By.ID, "txtDate") # Find the date input element
    date_input.clear() # Clear any existing value
    date_input.send_keys("01-01-2010") # Type the date (Selecting first available date)
    date_input.send_keys(Keys.ENTER) # Hit Enter
    time.sleep(3)

    logger.info("Submitting the form")
    # Print the current URL
    print("Final URL:", driver.current_url) #Comment this line if you don't want to see the URL
    # Wait for page load
    time.sleep(3)
    # Extract table data
    data = []
    table_rows = driver.find_elements(By.XPATH, '//*[@id="cphBody_GridPriceData"]/tbody/tr') # Locate table rows using XPath
    print(f"Number of rows found: {len(table_rows)}") # Print number of rows found
    logger.info(f"Number of rows found: {len(table_rows)}")
    # Iterate through rows and extract cell data
    logger.info("Extracting data from table, row by row, this may take a while...Please wait...")
    for rw in tqdm(table_rows, desc="Extracting rows", unit="row", ncols=100, colour="green"):
        cells = rw.find_elements(By.TAG_NAME, 'td')
        row_data = [cell.text.strip() for cell in cells]
        data.append(row_data)

    driver.quit() # Close the browser
    # Convert to DataFrame
    df = pd.DataFrame(data)
   
    logger.info("Data extraction complete.")
    # Export to Excel
    df.to_excel(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\Agriculture Price Prediction\Data\Raw\agmarknet_data new.xlsx", index=False)

    print("✅ Data exported to agmarknet_data.xlsx")