import streamlit as st
import pickle
import numpy as np


model = pickle.load(open("model/loan_model.pkl", "rb"))

st.set_page_config(page_title="Loan Approval System")
st.title(" Real-Time Loan Approval Prediction")
st.title("8 Legends Bank of MD College")


gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
married = st.selectbox("Married", ["Yes", "No"], key="married")
education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="education")
credit_history = st.selectbox("Credit History", [0, 1], key="credit")

income = st.number_input("Applicant Income", min_value=0, value=5000, key="income")
loan_amount = st.number_input("Loan Amount", min_value=0, value=2000, key="loan")


gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0

#
if st.button("Predict Loan Approval"):
    input_data = np.array([[gender, married, education, income, loan_amount, credit_history]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
