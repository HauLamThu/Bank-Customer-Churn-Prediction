import streamlit as st
import requests
from main import predict

st.title("🔍 Customer Churn Checker")

# 1. Input fields
Geography = st.text_input("Customer Name")
Age = st.slider("Age", 18, 80, 35)
Gender = st.selectbox("Gender", ["Male", "Female"])
Geography = st.selectbox("Geography", ["Germany", "France", 'Spain'])
Balance = st.number_input("Balance", 0, 200000, 5000)
Tenure = st.slider("Tenure (years)", 0, 10, 2)
IsActiceMember = st.selectbox("Is Active Member?", ["Yes", "No"])
NumOfProducts = st.slider("Number of Products",0, 4, 2)

# 2. Trigger model
if st.button("Check Churn Risk"):
    with st.spinner("Sending to your model..."):
        data = {
            "Geography": Geography,
            "Age": Age,
            "Gender": Gender,
            "Balance": Balance,
            "Tenure": Tenure,
            "IsActiceMember": 1 if IsActiceMember == "Yes" else 0,
            "NumOfProducts": NumOfProducts
        }

        try:
            
            pred_class, pred_prob = predict(data)
            st.write(f"Churn Probability: {pred_prob:.2%}")

            if pred_class == 1:
                st.error("⚠️ This customer is likely to CHURN.")
            elif pred_class == 0:
                st.success("✅ This customer is likely to STAY.")
            else:
                st.warning(pred_class)

        
        except Exception as e:
            st.error(f"❌ Could not reach model: {e}")
