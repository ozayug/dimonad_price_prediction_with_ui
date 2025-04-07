import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import RobustScaler

# Load the model
with open('modelcopy.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler – this line should *always* be above
with open('scalercopy.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit title
st.title("💎 Diamond Price Predictor")

# Input fields
carat = st.number_input("Carat", value=0.0)
color = st.number_input("Color (encoded)", value=0)
clarity = st.number_input("Clarity (encoded)", value=0)
depth = st.number_input("Depth", value=0.0)
table = st.number_input("Table", value=0.0)
x = st.number_input("X (length in mm)", value=0.0)
y = st.number_input("Y (width in mm)", value=0.0)
z = st.number_input("Z (depth in mm)", value=0.0)
cut_good = st.number_input("Cut (Good)", value=0)
cut_ideal = st.number_input("Cut (Ideal)", value=0)
cut_premium = st.number_input("Cut (Premium)", value=0)
cut_very_good = st.number_input("Cut (Very Good)", value=0)

# Prediction button
if st.button("Predict Price"):
    # Validation: check for zero or missing values
    if (carat == 0 or color == 0 or clarity == 0 or depth == 0 or table == 0 or
        x == 0 or y == 0 or z == 0 or
        (cut_good == 0 and cut_ideal == 0 and cut_premium == 0 and cut_very_good == 0)):
        
        st.warning("⚠️ Please fill in all fields with non-zero values before predicting.")
    else:
        # Prepare features
        input_features = pd.DataFrame([[carat, color, clarity, depth, table, x, y, z,
                                        cut_good, cut_ideal, cut_premium, cut_very_good]],
                                      columns=['carat', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z',
                                               'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good'])

        # Apply scaling
        input_scaled = scaler.transform(input_features[["carat"]])
        input_features[['carat']] = input_scaled

        # Make prediction using the model
        prediction = model.predict(input_features)

        # Display the result
        st.success(f"💰 Estimated Price: ${float(prediction[0]):,.2f}")

# Footer
st.markdown("---")
st.markdown("**Developed By :- Manthan Makani & Yug Oza**")
