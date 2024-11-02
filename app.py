import streamlit as st
import numpy as np
import pickle
import requests
import io

# Load the trained model and scaler
model = joblib.load('price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and input fields
st.title("House Price Prediction App")
st.write("Enter the details to predict the house price")
Avg_Area_Income = st.number_input("Average Area Income", min_value=0.0)
Avg_Area_House_Age = st.number_input("Average Area House Age", min_value=0.0)
Avg_Area_Number_of_Rooms = st.number_input("Average Area Number of Rooms", min_value=0.0)
Avg_Area_Number_of_Bedrooms = st.number_input("Average Area Number of Bedrooms", min_value=0.0)
Area_Population = st.number_input("Area Population", min_value=0.0)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms,
                            Avg_Area_Number_of_Bedrooms, Area_Population]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Price: ${prediction[0]:.2f}")
