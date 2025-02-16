import os
import pandas as pd
import dill
from lifetimes import GammaGammaFitter
from lifetimes.plotting import *
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(base_path, "dataset")
model_path = os.path.join(base_path, "model")
df = pd.read_csv(os.path.join(dataset_path, "clean_dataset.csv"))

last_order_date = df['transaction_date'].max()

data = summary_data_from_transaction_data(df, 'cb_customer_id', 'transaction_date', 'amount_after_discount',
                                          observation_period_end=last_order_date, freq="D")

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(data['frequency'], data['recency'], data['T'])
data['probability_alive'] = bgf.conditional_probability_alive(data['frequency'], data['recency'], data['T'])
summary_cal_holdout = calibration_and_holdout_data(df, 'cb_customer_id', 'transaction_date',
                                        calibration_period_end='2016-08-18',
                                        observation_period_end=last_order_date )
with open(os.path.join(model_path, "bgf_model.pkl"), 'wb') as file:
    dill.dump(bgf, file)

bgf.fit(summary_cal_holdout['frequency_cal'], summary_cal_holdout['recency_cal'],
        summary_cal_holdout['T_cal'])

returning_customers_summary = data[(data['frequency']>0.0)&(data['monetary_value']>0)]

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])

cltv = ggf.customer_lifetime_value(bgf,
                                   data['frequency'],
                                   data['recency'],
                                   data['T'],
                                   data['monetary_value'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv = data.merge(cltv, on="cb_customer_id", how='left').reset_index(drop = True)

with open(os.path.join(model_path, "ggf_model.pkl"), 'wb') as file:
    dill.dump(bgf, file)

cltv.to_csv(os.path.join(dataset_path, "cltv.csv"), index=False)