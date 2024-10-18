# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Diabetes Prediction App')

# Load the diabetes dataset
Diab_Data = pd.read_csv("diabetes.csv")

# Split data into features and target
X = Diab_Data.drop(columns=["Outcome"])
Y = Diab_Data["Outcome"]

# Split the data into training and testing sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy of the model
pred = model.predict(X_test)
accuracy = accuracy_score(pred, Y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")


# We Will take User input And Store It In A list For Prediction:
# User input for prediction
st.sidebar.header("Input Features")
st.sidebar.subheader("Enter Patient Data:")
Pregnancies = st.sidebar.slider("Pregnancies", 0, 20, step=1)
Glucose = st.sidebar.slider("Glucose Level", 0, 200, step=1)
BloodPressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, step=1)
SkinThickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, step=1)
Insulin = st.sidebar.slider("Insulin (mu U/ml)", 0, 900, step=1)
BMI = st.sidebar.slider("BMI", 0.0, 70.0, step=0.1)
DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
Age = st.sidebar.slider("Age", 0, 120, step=1)


# When this Button Is Pressed it Will Run A loop To Predict According the to The Data Input From the user:
# Predict button:
if st.button("Predict"):
    # Create a NumPy array with user input
    user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Make a prediction
    prediction = model.predict(user_data)
    
    # Show the result
    if prediction[0] == 1:
        st.write("The model predicts that this person **has diabetes**.")
    else:
        st.write("The model predicts that this person **does not have diabetes**.") # This Will Show The The Prediction that Have Been Made:
        