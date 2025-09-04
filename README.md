# Bank-Customer-Churn-Prediction-with-Deployment

This project predicts whether a bank customer will leave (churn) based on their personal and account information. It helps banks identify at-risk customers and take early action to retain them.
The app is built with Streamlit, containerized using Docker for demonstration.


Link to my Kaggle notebook: https://www.kaggle.com/code/haulam234/bank-customer-churn-prediction

## 📊 Dataset

The dataset was obtained from Kaggle: https://www.kaggle.com/competitions/bank-customer-churn-prediction-challenge 

- Rows: 15,000 customers
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Target: `Exited` (1 = churned, 0 = stayed)

## 🔎 Project Steps

1. **Exploratory Data Analysis (EDA)**
   - Checked data distribution, missing values.
   - Categorical encoding (Gender, Geography).
   - Visualized customer churn patterns by categorical and numerical variables.
   - Detected outliers.

3. **Feature engineering**  
   - Generated possible relevant variables.
   - Checked correlation for feature selection.

4. **Modeling**
   - Trained baseline models: Random Forest, XGBoost, Logistic Regression.
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
   - Selected the final model and feature importance analysis.

5. **Conclusion**
   - The final model Random Forest can cover 81% of customers about to churn. Among 10 customers predicted churn, there are 7 customers will actually leave.
   - Found that features like Age, Number of Products and Balance have strong impact on churn probability.
   - Received the public score of 92%.

## ✅ Deployment with Docker and Streamlit
1. Containerized by using Docker:
   - Created main.py file for preprocessing data, loading model and running model.
   - Defined requirements.txt for installing necessary dependencies.
   - Wrote a Dockerfile to run requirements.txt, main.py and Streamlit.
   - Built and run Docker image:
```
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```
2. Displayed by Streamlit:
   - Built a user-friendly UI for customer input.
   - Displays prediction results in real time.
   - View the deployed model UI:
![Model](https://github.com/HauLamThu/Bank-Customer-Churn-Prediction/blob/main/Streamlit.png)

