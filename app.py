import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load('titanic_model.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.42, max_value=80.0, value=30.0)
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare Paid (in dollars)", min_value=0.0, value=32.0)  # Add this line

if st.button("Predict"):
    sex_encoded = sex_encoder.transform([sex])[0]
    features = np.array([[pclass, age, sex_encoded, sibsp, parch, fare]])  # Include Fare here
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.success("🎉 Survived")
    else:
        st.error("❌ Did not survive")
