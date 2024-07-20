import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

# Title
st.title("Heart Disease Prediction")

# Create three columns for the first row of inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=52)

with col2:
    sex = st.selectbox('Sex', [1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')

with col3:
    cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=0)

# Create three columns for the second row of inputs
col4, col5, col6 = st.columns(3)

with col4:
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=125)

with col5:
    chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0, max_value=600, value=212)

with col6:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [1, 0], format_func=lambda x: 'True' if x == 1 else 'False')

# Create three columns for the third row of inputs
col7, col8, col9 = st.columns(3)

with col7:
    restecg = st.number_input('Resting Electrocardiographic Results ', min_value=0, max_value=2, value=1)

with col8:
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=168)

with col9:
    exang = st.selectbox('Exercise Induced Angina', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Create three columns for the fourth row of inputs
col10, col11, col12 = st.columns(3)

with col10:
    oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0)

with col11:
    slope = st.number_input('Slope of the peak exercise ST segment ', min_value=0, max_value=2, value=2)

with col12:
    ca = st.number_input('major vessels colored by fluoroscopy ', min_value=0, max_value=4, value=2)

# Create a single column for the last input
col13, col14 = st.columns(2)

with col13:
    thal = st.number_input('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)', min_value=1, max_value=3, value=3)

# Prediction
if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = sc.transform(input_data)
    prediction = classifier.predict(input_data_scaled)
    
    if prediction == 0:
        st.write("Prediction: No Heart Disease")
    else:
        st.write("Prediction: Heart Disease Detected")
