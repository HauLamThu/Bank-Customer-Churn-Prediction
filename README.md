# Bank-Customer-Churn-Prediction

This project aims to predict whether a bank customer is likely to churn (leave the bank) based on their profile and activity data. The goal is to help financial institutions proactively identify at-risk customers and improve customer retention strategies.

## 📊 Dataset

The dataset was obtained from Kaggle: https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge 

- Rows: 15,000 customers
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
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

5. **Conclusion**
   - Final model Random Forest can cover 81% of customers about to churn and among 10 customers predicted churn, there are 7 customers will actually leave
   - Found that features like Age, Number of Products and Balance have strong impact on churn probability
   - Received the public score with 92%

## ✅ Deployment with Docker, AWS and Streamlit
1. Docker
   - Prepared files and dependencies
   - Containerized the Streamlit app
   - Built and ran Docker image locally
2. AWS
   - Pushed Docker image to ECR
   - Deployed using ECS
3. Streamlit:
   - Built a simple UI for customer input
   - Displays prediction result in real time

<img width="780" alt="Screenshot 2025-07-08 at 00 03 49" src="https://github.com/user-attachments/assets/e555d929-86c7-4290-968f-baca9e61fcdc" />

