import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(base_path, "dataset")
df = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
df = df.drop_duplicates(keep = 'first').reset_index(drop = True)
df = df.drop(columns=["date_of_birth", "gender", "unique_customer_id"], axis=1).reset_index(drop = True)
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d")
df["Day"] = df["transaction_date"].dt.day
df["Month"] = df["transaction_date"].dt.month
df.to_csv(os.path.join(dataset_path, "clean_dataset.csv"), index=False)
