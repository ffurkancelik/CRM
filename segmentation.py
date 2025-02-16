import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dill
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(base_path, "dataset")
model_path = os.path.join(base_path, "model")
rfm_extended = pd.read_csv(os.path.join(dataset_path, "rfm_extended.csv"))
cltv = pd.read_csv(os.path.join(dataset_path, "cltv.csv"))


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv[["clv"]])
cltv["scaled_clv"] = scaler.transform(cltv[["clv"]])

cltv["segment"] = pd.qcut(cltv["scaled_clv"], 5, labels=["E","D", "C", "B", "A"])

with open(os.path.join(model_path, "scaler_segmentation.pkl"), 'wb') as file:
    dill.dump(scaler, file)

cltv.to_csv(os.path.join(dataset_path, "cltv_with_segments.csv"), index=False)


features = ['Recency', 'Frequency', 'Monetary']
X = rfm_extended[features]

k_means = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm_extended['Segment'] = k_means.fit_predict(X)

with open(os.path.join(model_path, "k_means_segmentation.pkl"), 'wb') as file:
    dill.dump(k_means, file)

rfm_extended.to_csv(os.path.join(dataset_path, "rfm_extended_with_segments.csv"), index=False)