# CLV & Churn & Customer Segmentation

## Data Analysis

### **Dataset**
- Initially, the following datasets were merged based on ID, and duplicate records were cleaned:
- `cust_best_sample.csv`, `cust_sample.csv`, `trx_sample.csv`.

### **Data Analysis**
- Basic statistics were calculated.
- Birth year and gender data were analyzed in relation to the rest of the dataset. Due to their high level of missing values and lack of correlation with other features, they were removed.

Birth Data:
- When the entire dataset was considered, birth years and consequently age data were 97% incomplete. Despite this, the relationship with the rest of the data was examined, and as seen in the last pair plot, no connection was found. Additionally, the presence of negative age values reduced confidence in the data.

Gender Data:
- When the entire dataset was considered, gender data was 96% missing. However, when analyzed for its correlation with other data points, no significant connection was observed, as shown in the last pair plot.

Correlation Between Birth and Gender Data:
- Given the above analysis and the rarity of both gender and birth data, this data was deemed insufficient for analysis due to it being present in only 90 different customers. Additionally, no periodic correlation between age and gender was found.

--------------------------------------------------------------
- TRX day and month information was created.
- Duplicate records were removed.
- Exploratory Data Analysis (EDA) was conducted on the dataset.

## Modeling

### **CLV (Customer Lifetime Value) Model**

#### **Frequency, T, Recency**
- Recency, Frequency, and Monetary (RFM) values were calculated to measure customer loyalty.

#### **BGF (Beta-Geometric Model)**
- Used to predict customer loyalty.
- Estimated the probability of future purchases.
- Calculated customer loyalty scores using the model.
- Determined whether a customer is still active.
- Made future purchase predictions.
- Evaluated model performance and suitability.

- Customers with low purchase frequency were filtered.

#### **GGF (Gamma-Gamma Model)**
- Used to predict transaction values.
- Customer profitability was calculated.

#### **CLV**
- Customer Lifetime Value was computed.

### **Churn (Customer Attrition) Model**

- Different methods were used to calculate RFM.
- Churn risk scores were calculated.
- High-risk customers were identified.
- Two different methods were used to detect churn customers.

#### **Churn Prediction - Supervised**
- Customers were labeled using churn rules, and a dataset was created.
- Logistic regression was used to predict customer attrition.

Churn Rules:
  - If the customer has not made any transactions in the last 60 days (Recency > 60), consider them churned.
  - If the customer has made only one transaction in six months (Low_Activity_Customer = 1), consider them churned.
  - If the customer has not spent anything in six months (Monetary = 0), consider them churned.

#### **Churn Prediction - Unsupervised**
- Anomaly detection was conducted using Isolation Forest.


- The results obtained using churn rules were more satisfactory than unspervised method.

### **Customer Segmentation**

#### **With CLV-Based Segmentation**
- Customer segmentation was performed based on the CLV model.

#### **K-Means with RFM (Recency, Frequency, Monetary)**
- Customers were clustered based on RFM values using the K-Means algorithm.

## Results and Outcomes
- **CLV Calculation**: Future purchase probabilities, profits, and customer lifetime values were estimated.
- **Churn Prediction**: Churn risks were calculated. Both supervised (rule-based) and unsupervised (anomaly detection) approaches were implemented.
- **Segmentation Results**: Customers were segmented based on both CLV values and the RFM method.

## Development Environment
1. Developed on Google Colab.
2. You can upload and run `crm.ipynb` on Colab or open it in Jupyter Notebook.

## Assumptions
- Due to the significant missing values in gender and birth year data, their impact could not be conclusively determined.
- Since the dataset only covers a 5-month period, seasonality trends could not be identified.
- As no predefined churn rules were provided, appropriate rules were applied based on the dataset.

## Model Outputs
- `bgf_model.pkl`: Beta-Geometric model
- `ggf_model.pkl`: Gamma-Gamma model
- `logistic_reg_churn.pkl`: Churn prediction model
- `k_means_segmentation.pkl`: K-Means segmentation model
- `IsolationForest_churn.pkl`: Anomaly detection model

# Additional Information
- Various analysis images can be found in the ipyn file and the additional images folder.
- The Code part also explained in the ipyn file.
- You can find some visual outputs of analysis and model results in the images folder.
- Raw and processed datasets can be found in the dataset folder. 
- Churn outputs stores inside the "rfm_extended.csv"
- All the process can be found in the ipyn file.
- There is only training scripts in the py files.
- You can train clv, churn and segmentatin with train.py

