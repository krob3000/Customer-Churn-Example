
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# Sidebar: Upload CSV
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.sidebar.title("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

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

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # -------------------------------
    # Model Performance
    # -------------------------------
    st.subheader("Model Performance")
    accuracy = model.score(X_test, y_test)
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Feature Importance
    # -------------------------------
    st.subheader("Top Drivers of Churn")
    coeffs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=coeffs.values, y=coeffs.index, palette='viridis', ax=ax)
    ax.set_title('Top Drivers of Churn')
    st.pyplot(fig)

    # -------------------------------
    # Recommendations
    # -------------------------------
    st.subheader("✅ Recommendations to Reduce Churn")
    st.markdown("""
    1. **Improve Customer Support** – Reduce resolution time and handle escalations proactively.
    2. **Enhance Engagement** – Increase monthly logins and weekly active days with loyalty programs.
    3. **Address Payment Issues** – Implement automated retry and flexible payment options.
    4. **Monitor Price Sensitivity** – Communicate value clearly when increasing prices.
    5. **Boost Satisfaction & NPS** – Act on feedback and improve product experience.
    """)
else:
    st.info("Please upload a CSV file to start.")
