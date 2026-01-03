import opendatasets as od

# The medical dataset link
dataset_url = "https://www.kaggle.com/datasets/paultimothymooney/kermany2018"

print("--- SYSTEM CHECK ---")
print("If you don't see a prompt, please wait 5 seconds...")

# This command MUST trigger the User Name prompt
od.download(dataset_url)

print("--- SUCCESS ---")