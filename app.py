
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# -------------------------------
# PDF Helper Functions
# -------------------------------

def draw_chart(canvas, fig, x=None, y=500, max_width=500, max_height=200, page_width=letter[0]):
    """Draw a matplotlib figure on the PDF canvas."""
    if fig is None:
        return y

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=150)
    buf.seek(0)

    img = ImageReader(buf)
    img_width, img_height = img.getSize()

    scale = min(max_width / img_width, max_height / img_height)
    draw_width = img_width * scale
    draw_height = img_height * scale

    if x is None:
        x = (page_width - draw_width) / 2

    canvas.drawImage(img, x, y, width=draw_width, height=draw_height)
    return y - draw_height - 30


def ensure_page_space(canvas, y, min_y=100):
    """Ensure enough space on the page; create a new page if needed."""
    if y < min_y:
        canvas.showPage()
        canvas.setFont("Helvetica", 12)
        return 750
    return y

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.sidebar.title("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

MODEL_FILE = "trained_churn_model.pkl"
SCALER_FILE = "scaler.pkl"

# Initialize chart variables
fig_cat = None
fig_roc = None
fig_feat = None

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
        fig_cat, ax = plt.subplots(figsize=(6, 4))
        churn_by_cat.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_ylabel('Churn Rate')
        ax.set_title(f'Churn Rate by {selected_col}')
        st.pyplot(fig_cat)

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
