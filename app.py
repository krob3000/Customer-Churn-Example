
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import io

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("Filters")
    if "Country" in df.columns:
        country_filter = st.sidebar.multiselect("Select Country", options=df["Country"].unique())
        if country_filter:
            df = df[df["Country"].isin(country_filter)]

    # -------------------------------
    # Feature Selection
    # -------------------------------
    st.subheader("Model Training")
    target_column = st.selectbox("Select Target Column (Churn)", options=df.columns)
    feature_columns = st.multiselect("Select Feature Columns", options=[col for col in df.columns if col != target_column])

    if st.button("Train Model"):
        if target_column and feature_columns:
            X = df[feature_columns]
            y = df[target_column]

            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Logistic Regression Model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # -------------------------------
            # Confusion Matrix
            # -------------------------------
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

            # -------------------------------
            # ROC Curve
            # -------------------------------
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            st.subheader("ROC Curve")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("Receiver Operating Characteristic")
            ax2.legend(loc="lower right")
            st.pyplot(fig2)

            # -------------------------------
            # Feature Importance
            # -------------------------------
            st.subheader("Feature Importance")
            feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
            ax3.set_title("Feature Importance (Logistic Regression Coefficients)")
            ax3.set_xlabel("Coefficient Value")
            st.pyplot(fig3)

            st.success("✅ Model training complete!")

            # -------------------------------
            # Auto-Save to Folder
            # -------------------------------
            save_folder = "saved_results"
            os.makedirs(save_folder, exist_ok=True)

            # Save model
            model_path = os.path.join(save_folder, "trained_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save predictions
            results_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Probability": y_prob
            })
            predictions_path = os.path.join(save_folder, "predictions.csv")
            results_df.to_csv(predictions_path, index=False)

            st.info(f"✅ Files saved automatically in: {os.path.abspath(save_folder)}")

            # -------------------------------
            # Download Buttons
            # -------------------------------
            st.subheader("Download Results")

            # Download model
            model_bytes = pickle.dumps(model)
            st.download_button(
                label="⬇ Download Trained Model",
                data=model_bytes,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

            # Download predictions
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇ Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )

        else:
            st.error("Please select target and feature columns.")
else:
    st.info("Upload a CSV file to start.")
