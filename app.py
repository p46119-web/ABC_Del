
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    lr_model = joblib.load('logi.sav')
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error('Error: logi.sav not found. Please ensure the model file is in the same directory.')
    st.stop()

# Define the exact feature names and order as used in X_encoded during training
# This is crucial for consistent input to the model
FEATURE_COLUMNS = [
    'delivery_distance',
    'traffic_congestion',
    'driver_experience',
    'num_stops',
    'vehicle_age',
    'road_condition_score',
    'package_weight',
    'fuel_efficiency',
    'warehouse_processing_time',
    'weather_condition_2',
    'weather_condition_3',
    'delivery_slot_2',
    'delivery_slot_3'
]

# Streamlit UI
st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input widgets for numerical features
delivery_distance = st.slider('Delivery Distance (km)', 0, 100, 20)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', 1, 5, 3)
driver_experience = st.slider('Driver Experience (years)', 0, 30, 5)
num_stops = st.slider('Number of Stops', 0, 20, 5)
vehicle_age = st.slider('Vehicle Age (years)', 0, 20, 3)
road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', 1, 5, 4)
package_weight = st.slider('Package Weight (kg)', 0, 50, 10)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', 5, 30, 15)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 200, 60)

# Input widgets for categorical features (one-hot encoded)
weather_condition = st.selectbox('Weather Condition', options=[1, 2, 3], format_func=lambda x: {1: 'Clear', 2: 'Rainy', 3: 'Snowy'}[x])
delivery_slot = st.selectbox('Delivery Slot', options=[1, 2, 3], format_func=lambda x: {1: 'Morning', 2: 'Afternoon', 3: 'Evening'}[x])

if st.button('Predict Delivery Delay'):
    # Prepare input data for the model
    input_data = {
        'delivery_distance': delivery_distance,
        'traffic_congestion': traffic_congestion,
        'driver_experience': driver_experience,
        'num_stops': num_stops,
        'vehicle_age': vehicle_age,
        'road_condition_score': road_condition_score,
        'package_weight': package_weight,
        'fuel_efficiency': fuel_efficiency,
        'warehouse_processing_time': warehouse_processing_time,
        'weather_condition_2': 1 if weather_condition == 2 else 0,  # One-hot encode weather_condition
        'weather_condition_3': 1 if weather_condition == 3 else 0,
        'delivery_slot_2': 1 if delivery_slot == 2 else 0,      # One-hot encode delivery_slot
        'delivery_slot_3': 1 if delivery_slot == 3 else 0
    }

    # Create a DataFrame from the input data, ensuring correct column order
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    # Make prediction
    prediction = lr_model.predict(input_df)
    prediction_proba = lr_model.predict_proba(input_df)[:, 1]

    st.subheader('Prediction Results:')
    if prediction[0] == 1:
        st.error(f'**Predicted: DELAYED** (Probability of delay: {prediction_proba[0]:.2f})')
    else:
        st.success(f'**Predicted: ON-TIME** (Probability of delay: {prediction_proba[0]:.2f})')

    st.write('---')
    st.write('Input Features for Prediction:')
    st.json(input_data)

