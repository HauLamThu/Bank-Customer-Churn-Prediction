# Bank-Customer-Churn-Prediction

This project aims to predict whether a bank customer is likely to churn (leave the bank) based on their profile and activity data. The goal is to help financial institutions proactively identify at-risk customers and improve customer retention strategies.

## 📊 Dataset

The dataset was obtained from Kaggle: https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge 

- Rows: ~10,000 customers
- Features: Age, Balance, CreditScore, Gender, Geography, Tenure, IsActiveMember, 
- Target: `Exited` (1 = churned, 0 = stayed)

## 🔎 Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Checked data distribution, missing values
   - Categorical encoding (e.g., Gender, Geography)
   - Visualized customer churn patterns by categorical and numerical variables
   - Feature scaling (e.g., StandardScaler)
   - Detected outliers

3. **Feature engineering**  
   - Generated possible relevant variables
   - Checked correlation for feature selection

4. **Modeling**
   - Trained baseline models: Random Forest, XGBoost, Logistic Regression
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Selected final model and feature importance analysis

5. **Insights**
   - Model can have 81% of catching customers about to churn and among 10 customers predicted churn, there are 7 customers will leave
   - Found that features like Age, Number of Products and Balance have strong impact on churn probability
