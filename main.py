import pandas as pd
import joblib
import numpy as np

# Load the model
best_model = joblib.load('best_model.joblib')


def preprocessing(X):
    expected_columns = [
        'Age', 'NumOfProducts', 'Geography_Germany', 'IsActiveMember',
        'Tenure_Products_Ratio', 'Gender', 'Age_Tenure_Ratio', 'LogBalance',
        'Geography_France'
    ]

    # Feature engineering
    if 'Gender' in X.columns:
        X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

    if 'Geography' in X.columns:
        X['Geography_France'] = (X['Geography'] == 'France').astype(int)
        X['Geography_Germany'] = (X['Geography'] == 'Germany').astype(int)

    if 'Tenure' in X.columns and 'NumOfProducts' in X.columns:
        X['Tenure_Products_Ratio'] = X.apply(lambda x: (x['Tenure'] + 1) / x['NumOfProducts'] if x['NumOfProducts'] != 0 else 0, axis=1)

    if 'Age' in X.columns and 'Tenure' in X.columns:
        X['Age_Tenure_Ratio'] = X['Age'] / (X['Tenure'] + 1)

    if 'Balance' in X.columns:
        X['LogBalance'] = np.log(X['Balance'] + 1)

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    # Return only the expected columns in the correct order
    return X[expected_columns]

def predict(data:dict):

    features = pd.DataFrame([data])
    # Preprocess the input data
    processed_data = preprocessing(features)
    # Predict the class
    pred_class = best_model.predict(processed_data).astype(int)[0]
    pred_prob = best_model.predict_proba(processed_data).astype(float)[0][1]
    return pred_class, pred_prob
