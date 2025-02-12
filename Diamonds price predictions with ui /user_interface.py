import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('diamonds.csv')

# Encode categorical features
Le = LabelEncoder()
Ohe = OneHotEncoder(sparse_output=False, drop='first')

df['color'] = Le.fit_transform(df['color'])
df['clarity'] = Le.fit_transform(df['clarity'])

encoded_features = Ohe.fit_transform(df[['cut']])
new_columns = Ohe.get_feature_names_out(['cut'])
df_encoded = pd.DataFrame(encoded_features, columns=new_columns)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns='cut', axis=1, inplace=True)

# Extract features (X) and target variable (y)
X = df[['carat', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Diamond Price Predictor")

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

# Make prediction
if st.button("Predict Price"):
    input_features = np.array([[carat, color, clarity, depth, table, x, y, z, cut_good, cut_ideal, cut_premium, cut_very_good]])
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)
    st.success(f"Predicted Price: ${prediction[0]:.2f}")
