import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from pickle import load
import streamlit as st

import joblib
import os

basedir = os.path.abspath(os.path.dirname(__file__))

model = joblib.load(os.path.join(basedir, "../models/arbol_de_decision_5_4_2_42.sav"))
scaler = joblib.load(os.path.join(basedir, "../models/scaler.save"))
print(type(scaler))

class_dict = {
    "0": "No diabetico",
    "1": "Diabetico",
}

st.title("Predicción de diabetes con Árbol de Decisión")

val1 = st.slider("Pregnancies", min_value = 0.0, max_value = 17.0, step = 1.0)
val2 = st.slider("Glucose", min_value = 45.0, max_value = 215.0, step = 1.0)
val3 = st.slider("SkinThickness", min_value = 5.0, max_value = 70.0, step = 0.1)
val4 = st.slider("Insulin", min_value = 30.0, max_value = 400.0, step = 1.0)
val5 = st.slider("BMI", min_value = 17.0, max_value = 60.0, step = 0.5)
val6 = st.slider("DiabetesPedigreeFunction", min_value = 0.0, max_value = 2.5, step = 0.02)
val7 = st.slider("Age", min_value = 20.0, max_value = 80.0, step = 1.0)

data = [[val1, val2, val3, val4, val5, val6, val7]]

scaled_data = scaler.transform(data)

if st.button("Predict"):
    prediction = str(model.predict(scaled_data)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)