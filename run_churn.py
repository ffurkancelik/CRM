import os
import numpy as np
import pandas as pd
import dill
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "model")
dataset_path = os.path.join(base_path, "dataset")
df = pd.read_csv(os.path.join(dataset_path, "clean_dataset.csv"))
df['cb_customer_id'] = df['cb_customer_id'].astype('int')
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
last_order_date = df['transaction_date'].max()

rfm_extended: DataFrame = df.groupby('cb_customer_id').agg(
    First_Purchase=('transaction_date', 'min'),
    Last_Purchase=('transaction_date', 'max'),
    Frequency=('transaction_date', 'count'),
    Monetary=('amount_after_discount', 'sum'),
    Avg_Transaction_Value=('amount_after_discount', 'mean'),
    Total_Discount_Used=('amount_discount', 'sum'),
    Discount_Ratio=('amount_discount', lambda x: x.sum() / (x.sum() + df['amount_before_discount'].sum()))
).reset_index()

rfm_extended['Tenure'] = (rfm_extended['Last_Purchase'] - rfm_extended['First_Purchase']).dt.days
rfm_extended['Recency'] = (last_order_date - rfm_extended['Last_Purchase']).dt.days
rfm_extended['Avg_Days_Between_Transactions'] = rfm_extended['Tenure'] / rfm_extended['Frequency']
rfm_extended['Churn_Score'] = rfm_extended['Recency'] / (rfm_extended['Frequency'] + 1)
rfm_extended['Inactive_Days'] = (last_order_date - rfm_extended['Last_Purchase']).dt.days
rfm_extended['Low_Activity_Customer'] = np.where(rfm_extended['Frequency'] < 2, 1, 0)

rfm_extended['Churn'] = np.where((rfm_extended['Recency'] > 60) |
                                 (rfm_extended['Low_Activity_Customer'] == 1) |
                                 (rfm_extended['Monetary'] == 0), 1, 0)

features = ['Recency', 'Frequency', 'Monetary', 'Avg_Transaction_Value',
            'Discount_Ratio', 'Churn_Score', 'Inactive_Days', 'Low_Activity_Customer']

X = rfm_extended[features]
y = rfm_extended['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

with open(os.path.join(model_path, "logistic_reg_churn.pkl"), 'wb') as file:
    dill.dump(log_reg, file)

features_clustering = ['Recency', 'Frequency', 'Monetary']
X_clustering = rfm_extended[features_clustering]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm_extended['KMeans_Cluster'] = kmeans.fit_predict(X_clustering)

iso_forest = IsolationForest(contamination=0.15, random_state=42)
rfm_extended['Anomaly_Score'] = iso_forest.fit_predict(X_clustering)

with open(os.path.join(model_path, "IsolationForest_churn.pkl"), 'wb') as file:
    dill.dump(iso_forest, file)

rfm_extended.to_csv(os.path.join(dataset_path, "rfm_extended.csv"), index=False)
