import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    scaler = StandardScaler()
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[num_cols] = scaler.fit_transform(data[num_cols])
    if data['churn'].dtype == 'float64':
        data['churn'] = data['churn'].round().astype(int)
    return data