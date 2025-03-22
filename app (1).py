import streamlit as st
import pandas as pd
import pickle

# Load the model using pickle
with open('best_regression_model (1).pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title='Mobile Price Predictor')
st.title('📱 Mobile Price Prediction App')
st.write("### Enter mobile specifications to predict the price!")

# Input fields using expander
with st.expander("### 📦 Mobile Features"):
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16])
    rom = st.selectbox('ROM (GB)', [16, 32, 64, 128, 256, 512, 1024])
    mobile_size = st.slider('Mobile Size (inches)', 4.0, 7.0, 6.5)
    primary_cam = st.slider('Primary Camera (MP)', 5, 108, 50)
    selfi_cam = st.slider('Selfie Camera (MP)', 5, 50, 32)
    battery_power = st.slider('Battery Power (mAh)', 2000, 7000, 5000)

# Create input data
sample_input = pd.DataFrame({
    'RAM': [ram],
    'ROM': [rom],
    'Mobile_Size': [mobile_size],
    'Primary_Cam': [primary_cam],
    'Selfi_Cam': [selfi_cam],
    'Battery_Power': [battery_power]
})

# Predict and display results
if st.button('💡 Predict Price'):
    with st.spinner('Calculating... Please wait!'):
        predicted_price = model.predict(sample_input)
        st.success(f"💰 Estimated Mobile Price: ₹{predicted_price[0]:,.2f}")

# Additional info
st.markdown("---")
st.write("### 📊 Model Details")
st.info("This prediction uses a Random Forest model with optimized hyperparameters via Optuna.")
st.write("Try adjusting the specifications to see how it influences the predicted price.")
