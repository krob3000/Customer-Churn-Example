
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os
import io

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.sidebar.title("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

MODEL_FILE = "trained_churn_model.pkl"
SCALER_FILE = "scaler.pkl"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.title("📊 Customer Churn Dashboard")

    # -------------------------------
    # Filters
    # -------------------------------
    st.sidebar.subheader("Filters")
    country_filter = st.sidebar.multiselect("Select Country", options=df['country'].unique())
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['gender'].unique())

    filtered_df = df.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]

    # -------------------------------
    # Overview Metrics
    # -------------------------------
    st.subheader("Overview")
    churn_rate = filtered_df['churn'].mean()
    col1, col2 = st.columns(2)
    col1.metric("Overall Churn Rate", f"{churn_rate:.1%}")
    col2.metric("Total Customers", f"{len(filtered_df):,}")

    # -------------------------------
    # Churn by Category
    # -------------------------------
    st.subheader("Churn Rate by Category")
    categorical_cols = ['gender', 'country', 'city', 'payment_method']
    selected_col = st.selectbox("Select a categorical column", [col for col in categorical_cols if col in filtered_df.columns])
    if selected_col:
        churn_by_cat = filtered_df.groupby(selected_col)['churn'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_cat.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_ylabel('Churn Rate')
        ax.set_title(f'Churn Rate by {selected_col}')
        st.pyplot(fig)

    # -------------------------------
    # Prepare Data for Model
    # -------------------------------
    numeric_cols = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
                    'avg_session_time', 'features_used', 'usage_growth_rate', 'last_login_days_ago',
                    'support_tickets', 'avg_resolution_time', 'csat_score', 'escalations',
                    'email_open_rate', 'marketing_click_rate', 'nps_score', 'payment_failures',
                    'referral_count', 'total_revenue', 'monthly_fee']

    selected_cols = [col for col in numeric_cols + categorical_cols if col in filtered_df.columns]
    X = filtered_df[selected_cols]
    y = filtered_df['churn']
    X = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], drop_first=True)

 
    # -------------------------------
    # Load or Train Model with Retrain Button
    # -------------------------------
    retrain = st.sidebar.button("🔄 Retrain Model")

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and not retrain:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        st.success("✅ Loaded existing model and scaler.")
    else:
        st.info("Training model... Please wait.")
        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Save model and scaler
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(X.columns.tolist(), "model_columns.pkl")
        st.success("✅ Model trained and saved.")


       
  # -------------------------------
    # Model Performance Charts
    # -------------------------------
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig_roc)

    # Feature Importance
    coeffs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10)
    fig_feat, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=coeffs.values, y=coeffs.index, palette='viridis', ax=ax)
    ax.set_title('Top Drivers of Churn')
    st.pyplot(fig_feat)

    # -------------------------------
    # Generate PDF Report
    # -------------------------------
    if st.button("📄 Download Polished PDF Report"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "Customer Churn Report")
        c.setFont("Helvetica", 12)
        c.drawString(50, 720, f"Overall Churn Rate: {churn_rate:.1%}")
        c.drawString(50, 700, f"Total Customers: {len(filtered_df):,}")
        c.drawString(50, 670, "Key Recommendations:")
        recommendations = [
            "Improve Customer Support",
            "Enhance Engagement",
            "Address Payment Issues",
            "Monitor Price Sensitivity",
            "Boost Satisfaction & NPS"
        ]
        y = 650
        for rec in recommendations:
            c.drawString(70, y, f"- {rec}")
            y -= 20

        # Save charts as images and embed
        img_buffer_cat = io.BytesIO()
        if fig_cat:
            fig_cat.savefig(img_buffer_cat, format='PNG')
            img_buffer_cat.seek(0)
            c.drawImage(img_buffer_cat, 50, 400, width=250, height=150)

        img_buffer_roc = io.BytesIO()
        fig_roc.savefig(img_buffer_roc, format='PNG')
        img_buffer_roc.seek(0)
        c.drawImage(img_buffer_roc, 320, 400, width=250, height=150)

        img_buffer_feat = io.BytesIO()
        fig_feat.savefig(img_buffer_feat, format='PNG')
        img_buffer_feat.seek(0)
        c.drawImage(img_buffer_feat, 50, 200, width=500, height=150)

        c.showPage()
        c.save()
        buffer.seek(0)

        st.download_button(
            label="Download Report",
            data=buffer,
            file_name="Customer_Churn_Report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Please upload a CSV file to start.")
