import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(base_path, "dataset")
cbs = pd.read_csv(os.path.join(dataset_path, "cust_best_sample.csv"))
cs = pd.read_csv(os.path.join(dataset_path, "cust_sample.csv"))
ts = pd.read_csv(os.path.join(dataset_path, "trx_sample.csv"))
df = ts.merge(cs, on="cb_customer_id", how="left")
df = df.merge(cbs, on="unique_customer_id", how="left")
df.to_csv(os.path.join(dataset_path, "dataset.csv"), index=False)
