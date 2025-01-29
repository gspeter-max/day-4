
'''
Problem: Predictive Maintenance in IoT Sensors
Scenario:
You are a data scientist working for a manufacturing company. The company has IoT-enabled machinery that continuously sends data from multiple sensors (e.g., temperature, pressure, vibration, etc.) during operation. Your task is to predict machine failures before they occur.

The challenges include:

High-dimensional time-series data from hundreds of sensors across multiple machines.
The data is highly imbalanced, as failures are rare compared to normal operations.
Sensor readings often have missing values or noise.
Early identification of potential failures is crucial to avoid downtime or costly repairs.
''' 


import numpy as np
import pandas as pd

np.random.seed(42)

n_machines = 100   
n_sensors = 100    
time_steps = 10000  

n_records = n_machines * time_steps

sensor_data = np.random.normal(0, 1, size=(n_records, n_sensors))

n_failures = int(0.01 * n_records)  
failure_indices = np.random.choice(range(n_records), size=n_failures, replace=False)

for idx in failure_indices:
    sensor_data[idx] = sensor_data[idx] + np.random.normal(10, 5, size=n_sensors)

machine_ids = np.repeat(range(n_machines), time_steps)
timestamps = np.tile(range(time_steps), n_machines)

labels = np.zeros(n_records)
labels[failure_indices] = 1

df = pd.DataFrame(sensor_data, columns=[f"sensor_{i}" for i in range(n_sensors)])
df['machine_id'] = machine_ids
df['date'] = timestamps
df['failure'] = labels

''' some thinking 
 XGBoost is the most versatile model for this  scenario. It is highly suitable for large, high-dimensional, noisy, and imbalanced datasets ''' 

print(df.info())
print(df.isnull().sum())


''' if have np.nan values '''
# from sklearn.impute import SimpleImputer


# numerical_features = [f"sensor_{i}" for i in range(100)] + ['date']
# categorical_features = ['machine_id', 'failure']

# numerical_imputer = SimpleImputer(strategy = 'mean')
# categorical_imputer = SimpleImputer( strategy = 'most_frequent')

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import  Pipeline 

# numerical_Pipline = Pipeline([
# 		('imputer',SimpleImputer(strategy = 'mean')),
# 		('standard',StandardScaler())
# 	])

# categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'most_frequent'))
# 	,('One_hot', OneHotEncoder())])

# preprocessing = ColumnTransformer(
# 	transformers = [
# 		('num',numerical_Pipline, numerical_features),
# 		('cat',categorical_pipeline, categorical_features)
# 	]
# )

# preprocessing_df = preprocessing.fit_transform(df)

from sklearn.model_selection import train_test_split 
from imblearn.over_sampling  import SMOTE, RandomOverSampler 

x = df.drop(columns = ['failure', "date", "machine_id"])
y = df['failure']
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

smote = SMOTE(random_state = 42)
x_train_resample, y_train_resample = smote.fit_resample(x_train, y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train_resample)
x_test = scaler.fit_transform(x_test)
# why i am not using that pipeline to do that and why i am  not use that scaled data that i am use in above answer that ? if you == 'bast data scientist': 


from xgboost import XGBClassifier 
from sklearn.metrics import classification_report , confusion_matrix , roc_auc_score 

model =  XGBClassifier(
	n_estimators = 300,
	learning_rate = 0.34,
	random_state = 42 ,
	eval_metric = 'auc',
	max_depth = 5 ,
    n_jobs = 2
    
  
)	

model.fit(x_train_scaled,y_train_resample)
y_predict_proba = model.predict_proba(x_test)[:, 1]
y_predict = model.predict(x_test)

print(f" classification report : {classification_report(y_test, y_predict)}")
print(f" confusion matrix : {confusion_matrix(y_test, y_predict)}")
print(f"roc_auc_score : {roc_auc_score(y_test, y_predict_proba)}")


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 120, 300],
    'learning_rate': [0.1, 0.4, 0.2],
    'max_depth': [3, 5, 7]
}

# Define base model
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='auc',
    n_jobs=2,
    use_label_encoder=False,
    tree_method='hist'
)

# GridSearchCV with 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    verbose=2,
    scoring='roc_auc'
)

# Fit GridSearchCV
grid_search.fit(x_train_scaled, y_train_resample)

# Best model and parameters
print(f"Best Params: {grid_search.best_params_}")
best_estimator = grid_search.best_estimator_

# Predictions
best_y_predict = best_estimator.predict(x_test)
best_y_proba = best_estimator.predict_proba(x_test)[:, 1]  # FIXED: use predict_proba

# Evaluation Metrics
print("\nClassification Report:")
print(classification_report(y_test, best_y_predict))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_y_predict))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, best_y_proba))
