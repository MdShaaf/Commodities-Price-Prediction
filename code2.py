from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import pandas as pd
import time
import tempfile

# Create a temporary Chrome profile folder
temp_profile = tempfile.mkdtemp()

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument(f"--user-data-dir={temp_profile}")  # unique user data directory
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

# Launch Chrome
driver = webdriver.Chrome(options=chrome_options)

# Open Agmarknet
driver.get("https://agmarknet.gov.in/")

# ---- Select dropdown values ----
commodity = Select(driver.find_element("id", "ddlCommodity"))
commodity.select_by_visible_text("Ginger(Green)")

state = Select(driver.find_element("id", "ddlState"))
state.select_by_visible_text("Karnataka")

district = Select(driver.find_element("id", "ddlDistrict"))
district.select_by_visible_text("Bangalore")

market = Select(driver.find_element("id", "ddlMarket"))
market.select_by_visible_text("Bangalore")

# Enter date range
driver.find_element("id", "txtDate").clear()
driver.find_element("id", "txtDate").send_keys("01-Jan-2010")

driver.find_element("id", "txtDateTo").clear()
driver.find_element("id", "txtDateTo").send_keys("09-Sep-2025")

# Click search
driver.find_element("id", "btnGo").click()

time.sleep(5)  # wait for table to load

# ---- Extract results into pandas ----
tables = pd.read_html(driver.page_source)
df = tables[0]

# Save results
df.to_excel("agmarknet_data.xlsx", index=False)

print("Data saved to agmarknet_data.xlsx")
print(df.head())

driver.quit()
