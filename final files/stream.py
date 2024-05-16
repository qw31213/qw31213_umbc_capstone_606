#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the preprocessing pipeline and model
preprocessing_pipeline = joblib.load('preprocessing_pipeline.joblib')
model = joblib.load('gaussian_process_model.joblib')

# Create a Streamlit app
st.title("Heart Murmur Prediction App")
with st.form(key='input_form'):

    # Get user input
    Recording_locations = st.selectbox("Recording locations:", ["AV+PV+TV+MV", "AV+MV", "PV+TV+MV", "AV", "MV", "AV+PV+MV", "PV+MV", "AV+AV+PV+PV+TV+MV", "AV+PV+TV", "AV+PV+TV+MV+MV", "PV", "PV+TV+TV", "AV+PV+PV+TV+TV+MV", "PV+TV", "AV+MV+MV", "AV+PV+MV+Phc+Phc", "AV+PV", "TV", "AV+AV+MV", "AV+PV+TV+TV+MV", "AV+AV+MV+MV", "AV+AV+PV+TV+MV", "AV+TV+MV", "AV+PV+TV+MV+Phc", "AV+AV+AV+MV", "AV+AV+PV+TV+MV+MV", "TV+MV", "TV+MV+Phc"])
    Age = st.selectbox("Age:", ["Child", "Adolescent", "Infant","Neonate","N/A"])
    Sex = st.selectbox("Sex:", ["Male", "Female"])
    Murmur_locations = st.selectbox("Murmur locations:", ["MV+PV+TV", "AV+MV+PV+TV", "TV", "MV+TV", "MV", "MV+PV", "AV+MV", "AV+PV+TV", "AV+MV+PV", "AV+PV", "PV", "AV+MV+PV+Phc+TV", "PV+TV", "AV", "MV+Phc+TV", "AV+TV", "AV+MV+TV"])
    Pregnancy_status = st.selectbox("Pregnancy status:", ["False", "True"])
    Diastolic_murmur_grading = st.selectbox("Diastolic murmur grading:", ["I/IV", "II/IV", "III/IV"])
    Systolic_murmur_grading = st.selectbox("Systolic murmur grading:", ["I/IV", "II/IV", "III/IV"])
    Murmur = st.selectbox("Murmur:", ["Absent", "Present", "Unknown"])
    Most_audible_location = st.selectbox("Most audible location:", ["AV", "TV", "PV", "MV"])
    Systolic_murmur_timing = st.selectbox("Systolic murmur timing:", ["Holosystolic", "Mid-systolic", "Early-systolic","Late Systolic"])
    Systolic_murmur_pitch = st.selectbox("Systolic murmur pitch:", ["Low", "Medium", "High"])
    Diastolic_murmur_pitch = st.selectbox("Diastolic murmur pitch:", ["Low", "Medium", "High"])
    Height = st.number_input("Height:")
    Weight = st.number_input("Weight:")

    # Create a submit button
    submit_button = st.form_submit_button(label='Submit')

# Check if the submit button has been pressed
if submit_button:
    # Create a DataFrame with the user input
    data = pd.DataFrame({
        "Recording locations:": [Recording_locations],
        "Age": [Age],
        "Sex": [Sex],
        "Murmur locations": [Murmur_locations],
        "Pregnancy status": [Pregnancy_status],
        "Diastolic murmur grading": [Diastolic_murmur_grading],
        "Systolic murmur grading": [Systolic_murmur_grading],
        "Murmur": [Murmur],
        "Most audible location": [Most_audible_location],
        "Systolic murmur timing": [Systolic_murmur_timing],
        "Systolic murmur pitch": [Systolic_murmur_pitch],
        "Diastolic murmur pitch": [Diastolic_murmur_pitch],
        "Height": [Height] if Height != "" else 0,
        "Weight": [Weight] if Weight != "" else 0
    })


    input_data_encoded = pd.DataFrame(preprocessing_pipeline.transform(data))

    # Make a prediction using the model
    prediction = model.predict(input_data_encoded)

    # Display the predicted injury severity
    st.write(f'Predicted injury severity: {prediction[0]}')

