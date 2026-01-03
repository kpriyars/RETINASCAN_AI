import os
import opendatasets as od

# 1. HARDCODE YOUR KEYS HERE (Keep the quotation marks!)
os.environ['KAGGLE_USERNAME'] = "kpkrishnapriya28"
os.environ['KAGGLE_KEY'] = "5f57e2aac6166ae3f37e354bb41f2908"

# 2. The medical dataset link
dataset_url = "https://www.kaggle.com/datasets/paultimothymooney/kermany2018"

print("--- STARTING DIRECT DOWNLOAD ---")
print("No typing required. Please wait...")

try:
    od.download(dataset_url)
    print("--- SUCCESS: Data is downloading! ---")
except Exception as e:
    print(f"Error: {e}")