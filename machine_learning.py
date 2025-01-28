import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

np.random.seed(42)

n_transactions = 1000000
n_features = 50

normal_data = np.random.randn(n_transactions, n_features)

n_fraud = int(0.01 * n_transactions)
fraud_data = np.random.randn(n_fraud, n_features) * 5

data = np.vstack([normal_data, fraud_data])

labels = np.zeros(n_transactions)
labels = np.hstack([labels, np.ones(n_fraud)])

data, labels = shuffle(data, labels, random_state=42)

transaction_amount = np.random.uniform(1, 5000, size=n_transactions + n_fraud)
account_age = np.random.randint(1, 20, size=n_transactions + n_fraud)
transaction_time = np.random.randint(0, 24, size=n_transactions + n_fraud)

df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
df['transaction_amount'] = transaction_amount
df['account_age'] = account_age
df['transaction_time'] = transaction_time
df['label'] = labels

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(columns='label'))

columns = scaled_data.shape[1]
df_scaled = pd.DataFrame(scaled_data, columns=[f'feature_{i}' for i in range(columns)])
df_scaled['transaction_amount'] = df['transaction_amount']
df_scaled['account_age'] = df['account_age']
df_scaled['transaction_time'] = df['transaction_time']
df_scaled['label'] = df['label']


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import IsolationForest 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report , roc_auc_score 


x = df_scaled.drop(columns = 'label')
y = df_scaled['label']

x = x.fillna(x.mean())
# scaler = StandardScaler()
# scaler.fit_transform(x)

pca = PCA(n_components = 50)
x_pca = pca.fit_transform(x)

model = IsolationForest(contamination = 0.2)
model.fit(x_pca)

y_predict = model.predict(x_pca)
print(y_predict)
