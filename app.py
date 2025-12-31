
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Customer Churn Prediction")

# ✅ Load hardcoded CSV from repo
df = pd.read_csv("data.csv")  # Make sure data.csv is in the same folder
st.write("### Dataset Preview", df.head())

# Check if 'Churn' column exists
if "Churn" in df.columns:
    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Sidebar for user input
    st.sidebar.header("Enter Customer Details")
    input_data = {}
    for col in X.columns:
        input_data[col] = st.sidebar.number_input(col, value=float(X[col].mean()))

    # Predict
    if st.sidebar.button("Predict Churn"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"### Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
else:
    st.error("CSV must contain a 'Churn' column for prediction.")
``
