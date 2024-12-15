import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('diabetes_model_xgb.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function for predicting diabetes
def predict_diabetes(input_data):
    # Convert input data to a numpy array and reshape it
    input_data_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data_array)
    
    # Make predictions using the loaded model
    prediction = model.predict(input_data_scaled)
    proba = model.predict_proba(input_data_scaled)
    
    return prediction[0], proba[0]

# streamlit app
st.title("Diabetes Prediction App")

st.markdown("""
This app predicts whether a person has diabetes based on medical input data.
Please provide the required information below.
""")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, value=30.0)
bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, value=30)

# Prediction
if st.button("Predict"):
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    prediction, probability = predict_diabetes(input_data)
    
    if prediction == 0:
        st.success("The person is not diabetic.")
    else:
        st.error("The person is diabetic.")
    st.write(f"Probability of being diabetic: {probability[1]*100:.2f}%")
