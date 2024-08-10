import streamlit as st
import joblib
import numpy as np  # numpy as np olarak import edilmeli
import pandas as pd

scaler = joblib.load("scaler.pkl")  # sclaer yerine scaler
model = joblib.load("model.pkl")

st.title("Customer Car Price Estimator App")

st.divider()

st.write("""This app is for getting price estimation for the customer so 
        a car with the price range given can be advised to the customer""")

age = st.number_input("Enter the age")
salary = st.number_input("Enter the salary", min_value=1000, max_value=99999999)
networth = st.number_input("Enter the net worth", min_value=0, max_value=99999999, step=2000, value=100000)

X = (age, salary, networth)

calculatebutton = st.button("Calculate")

st.divider()

if calculatebutton:
  
  X_2 = np.array(X).reshape(1, -1)  # X'i 2D array'e dönüştürmek için reshape kullanın
  
  X_array = scaler.transform(X_2)  # scaler'ı doğru kullanın

  prediction = model.predict(X_array)

  st.write(f"Prediction is {prediction[0]:,.2f}")
  st.write("Advise cars in the values")
else:
  st.write("Please enter the values and press the calculate button")