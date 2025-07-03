from fastapi import FastAPI  
import pandas as pd
import joblib
import numpy as np
import pickle  
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()  
# Load the GridSearchCV object
best_model = joblib.load('best_model.joblib')

# Extract the best model (already refitted on full training data)
#best_model = grid_search_rf.best_estimator_

class_names = np.array([0, 1])

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

@app.get("/")
async def root():
    return {"message": "Welcome to the Bank Churn Prediction API!"}

@app.post("/predict")
def predict(data:dict):
    #features = np.array(data[features].reshape(1,-1))
    features = pd.DataFrame([data])
    # Preprocess the input data
    processed_data = preprocessing(features)
    # Predict the class
    prediction = "Churn" if best_model.predict(processed_data).astype(int)[0] == 1 else "Non-churn"
    return {"Prediction": prediction}



# Add a test endpoint to your FastAPI app
# @app.get("/test")
# async def test_endpoint():
#     test_data = {
#         "Geography": "France",
#         "Gender": "Male",
#         "Age": 40,
#         "Tenure": 3,
#         "Balance": 120000,
#         "NumOfProducts": 1,
#         "IsActiveMember": 1
#     }
#     # Call your prediction function here
#     # result = await predict(test_data)
#     return {"test_data": test_data, "message": "Modify this to call your prediction"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
