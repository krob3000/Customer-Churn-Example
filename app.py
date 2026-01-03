
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import numpy as np

# -------------------------------
# PDF Helper Functions
# -------------------------------
def draw_chart(pdf_canvas, fig, x=None, y=500, max_width=500, max_height=200, page_width=letter[0]):
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

    pdf_canvas.drawImage(img, x, y, width=draw_width, height=draw_height)
    return y - draw_height - 30

def ensure_page_space(pdf_canvas, y, min_y=100):
    """Ensure enough space on the page; create a new page if needed."""
    if y < min_y:
        pdf_canvas.showPage()
        pdf_canvas.setFont("Helvetica", 12)
        return 750
    return y

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.sidebar.title("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# -------------------------------
# Cache Model Training
# -------------------------------
@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, scaler

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
    # Train Model with Spinner
    # -------------------------------
    with st.spinner("Training model... This may take a few seconds"):
        model, scaler = train_model(X, y)
    st.success("✅ Model trained successfully.")

    # -------------------------------
    # Model Performance Charts
    # -------------------------------
    X_scaled = scaler.transform(X)
    y_pred_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig_roc)

    # -------------------------------
    # Feature Importance Chart
    # -------------------------------
    try:
        feature_importance = abs(model.coef_[0])
        feature_names = X.columns
        sorted_idx = np.argsort(feature_importance)[::-1][:10]
        fig_feat, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=feature_importance[sorted_idx], y=feature_names[sorted_idx], ax=ax)
        ax.set_title("Top 10 Features Impacting Churn")
        st.pyplot(fig_feat)
    except Exception:
        st.warning("Feature importance chart could not be generated.")

    # -------------------------------
    # Enhanced PDF Report
    # -------------------------------
    st.subheader("Download Enhanced Report")

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)

    # Title Page
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawCentredString(letter[0]/2, 700, "Customer Churn Analysis Report")
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(letter[0]/2, 660, "Generated by Streamlit App")
    pdf.showPage()

    # Overview Page
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 750, "Overview Metrics")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, 720, f"Overall Churn Rate: {churn_rate:.1%}")
    pdf.drawString(50, 700, f"Total Customers: {len(filtered_df):,}")
    pdf.showPage()

    # Charts Page
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 750, "Visual Analysis")
    y_pos = 700

    if fig_cat:
        pdf.setFont("Helvetica", 14)
        pdf.drawString(50, y_pos, "Churn Rate by Category")
        y_pos -= 30
        y_pos = draw_chart(pdf, fig_cat, y=y_pos)
        y_pos = ensure_page_space(pdf, y_pos)

    if fig_roc:
        pdf.setFont("Helvetica", 14)
        pdf.drawString(50, y_pos, "ROC Curve")
        y_pos -= 30
        y_pos = draw_chart(pdf, fig_roc, y=y_pos)

    if fig_feat:
        y_pos = ensure_page_space(pdf, y_pos)
        pdf.setFont("Helvetica", 14)
        pdf.drawString(50, y_pos, "Top 10 Features Impacting Churn")
        y_pos -= 30
        y_pos = draw_chart(pdf, fig_feat, y=y_pos)

    pdf.save()

    st.download_button(
        label="📥 Download Enhanced PDF Report",
        data=buf.getvalue(),
        file_name="enhanced_churn_report.pdf",
        mime="application/pdf"
    )
