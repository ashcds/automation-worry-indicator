# Import necessary libraries
import streamlit as st
import joblib as jl
import pandas as pd
import numpy as np

st.title("Automation Worry Predictor") # Title
st.markdown("""
                This application predicts the likelihood of individuals feeling worried about job automation based on their responses to a survey. 
                The model is trained on data obtained from PEW Research Center's survey on technology and automation (American Trends Panel Wave 27).
                """)

# Load the pre-trained model
model = jl.load('best_xgb_model.pkl')

# Input fields for user to enter data
st.header("Please answer the following questions:")
cars3b = st.selectbox(
        "How worried are you, if at all, about the development of driverless vehicles?", 
        ["--Select an Option--", "Very Worried", "Somewhat Worried", "Not Too Worried", "Not at All Worried"])

cars7a = st.selectbox(
        "How safe would you feel sharing the road with a driverless passenger vehicle?",
        ["--Select an Option--", "Very Safe", "Somewhat Safe", "Not Too Safe", "Not at All Safe"])

cars7b = st.selectbox(
        "How safe would you feel sharing the road with a driverless freight truck?", 
        ["--Select an Option--", "Very Safe", "Somewhat Safe", "Not Too Safe", "Not at All Safe"])

robjob4b = st.selectbox("If robots and computers were able to perform most of the jobs currently being done by humans, would it be likely or unlikely that inequality between rich and poor would be much worse than it is today?", 
                            ["--Select an Option--", "Yes Likely", "No, Not Likely"])

robjob4a = st.selectbox("If robots and computers were able to perform most of the jobs currently being done by humans, would it be likely or unlikely that people would have a hard time finding things to do with their lives?", 
                            ["--Select an Option--", "Yes Likely", "No, Not Likely"])

    # Convert input data to appropriate format for prediction
input_data = pd.DataFrame({
        'CARS3B_W27': [1 if cars3b == "Very Worried" else 2 if cars3b == "Somewhat Worried" else 3 if cars3b=="Not Too Worried"  else 4],
        'ROBJOB4B_W27': [1 if robjob4b == "Yes Likely" else 2],
        'CARS7B_W27': [1 if cars7b == "Very Safe" else 2 if cars7b == "Somewhat Safe" else 3 if cars7b=="Not Too Safe"  else 4],
        'ROBJOB4A_W27': [1 if robjob4a == "Yes Likely" else 2],
        'CARS7A_W27': [1 if cars7a == "Very Safe" else 2 if cars7a == "Somewhat Safe" else 3 if cars7a=="Not Too Safe"  else 4]
    })

    # Diplay prediction button and prediction result
if st.button("Predict"):
        prediction = model.predict(input_data)
        prob_worried = model.predict_proba(input_data)[0, 1] * 100
        prob_not_worried = 100 - prob_worried

        if prediction[0] == 1:
            st.success(f"Prediction: You are likely to be **worried** about job automation.")
        else:
            st.success(f"Prediction: You are likely to be **not worried** about job automation.")
