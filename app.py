import streamlit as st
import numpy as np
import joblib
import requests
import io

# Function to load model and scaler from GitHub
@st.cache
def load_model_and_scaler():
    model_url = "https://github.com/MMotaghianfar/streamlit_price_prediction_api/raw/main/price_prediction_model.pkl"
    scaler_url = "https://github.com/MMotaghianfar/streamlit_price_prediction_api/raw/main/scaler.pkl"

    # Download and load the model
    model_response = requests.get(model_url)
    model = joblib.load(io.BytesIO(model_response.content))

    # Download and load the scaler
    scaler_response = requests.get(scaler_url)
    scaler = joblib.load(io.BytesIO(scaler_response.content))

    return model, scaler

# Load the model and scaler
model, scaler = load_model_and_scaler()

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
    # Prepare the input data
    input_data = np.array([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms,
                            Avg_Area_Number_of_Bedrooms, Area_Population]])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction and display it
    prediction = model.predict(scaled_input)
    st.write(f"Predicted Price: ${prediction[0]:.2f}")
