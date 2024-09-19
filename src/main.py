
import os
from .data_utils import load_and_preprocess_data
from .model import train_and_select_best_model
from .explanations import get_shap_values, get_lime_explanation

# # Load and preprocess the data
# data1 = load_and_preprocess_data('C:/Users/User/Desktop/assignment/generic-churn-prediction/data//Bank_churn.csv')
# data2 = load_and_preprocess_data('../data/churn_dataset1.csv')
# data3 = load_and_preprocess_data('../data/churn_dataset1.csv')

data_dir = 'C:/Users/User/Desktop/assignment/generic-churn-prediction/data/'

# Iterate over each CSV file in the data directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        print(f"Processing {file_name}...")
        
        # Load and preprocess the dataset
        file_path = os.path.join(data_dir, file_name)
        data = load_and_preprocess_data(file_path)
        print(data.head())
        # Separate features and target
        X = data.drop('churn', axis=1)
        y = data['churn']
        
        # Train the model and select the best one
        best_model = train_and_select_best_model(X, y)
        
        # Generate SHAP values for model interpretation
        print(f"Generating SHAP values for {file_name}...")
        get_shap_values(best_model, X)
        
        # Generate a LIME explanation for a single instance
        sample_instance = X.iloc[0]  # Just an example
        print(f"Generating LIME explanation for {file_name}...")
        get_lime_explanation(best_model, X, sample_instance)
        
        print(f"Completed processing for {file_name}.\n")

print("All datasets processed.")

