
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import io
import numpy as np

# ReportLab for PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# -------------------------------
# Streamlit Page Config
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

# Recommendations
recommendations = [
    "Improve onboarding experience to reduce early churn.",
    "Offer personalized discounts or loyalty programs for at-risk customers.",
    "Enhance customer support responsiveness and resolution times.",
    "Monitor and optimize product usage patterns to increase engagement.",
    "Implement proactive outreach for customers showing declining activity."
]

# Initialize chart variables
fig_cat = None
fig_roc = None
fig_feat = None

# -------------------------------
# PDF Footer with Page Numbers
# -------------------------------
def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(letter[0] - 40, 20, text)
    canvas.drawString(40, 20, "Company Confidential")

# -------------------------------
# Modern Cover Page with Gradient Overlay
# -------------------------------
def draw_cover_page(canvas, doc, logo_path, bg_image_path=None):
    # Draw background image
    if bg_image_path:
        canvas.drawImage(bg_image_path, 0, 0, width=letter[0], height=letter[1])
    else:
        # Fallback background color
        canvas.setFillColor(colors.HexColor("#F0F4F8"))
        canvas.rect(0, 0, letter[0], letter[1], fill=1)

    # Gradient overlay for modern look
    canvas.saveState()
    for i in range(100):
        alpha = 0.7 - (i * 0.007)
        canvas.setFillColorRGB(1, 1, 1, alpha=alpha)
        canvas.rect(0, i * (letter[1] / 100), letter[0], letter[1] / 100, fill=1, stroke=0)
    canvas.restoreState()

    # Logo
    if logo_path:
        canvas.drawImage(logo_path, letter[0]/2 - inch, letter[1] - 3*inch, width=2*inch, height=2*inch)

    # Modern Typography for Title
    canvas.setFont("Helvetica-Bold", 36)
    canvas.setFillColor(colors.HexColor("#2E4053"))
    canvas.drawCentredString(letter[0]/2, letter[1]/2 + 80, "Customer Churn Analysis Report")

    # Subtitle
    canvas.setFont("Helvetica-Oblique", 20)
    canvas.setFillColor(colors.HexColor("#4B9CD3"))
    canvas.drawCentredString(letter[0]/2, letter[1]/2 + 40, "Insights • Predictions • Actions")

    # Edition info
    canvas.setFont("Helvetica", 14)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(letter[0]/2, letter[1]/2 - 20, "January 2026 Edition")

# -------------------------------
# Build Magazine-Style PDF
# -------------------------------
def build_magazine_pdf(churn_rate, filtered_df, fig_cat, fig_roc, fig_feat, recommendations, logo_path, bg_image_path):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    subtitle_style = ParagraphStyle('SubtitleStyle', fontSize=16, alignment=1, textColor=colors.HexColor('#4B9CD3'), spaceAfter=15)
    body_style = ParagraphStyle('BodyStyle', fontSize=12, leading=16, spaceAfter=12)

    elements = []

    # Table of Contents
    elements.append(Paragraph("<b>Table of Contents</b>", subtitle_style))
    toc_items = [
        "1. Summary",
        "2. Overview Metrics",
        "3. Visual Analysis",
        "4. Recommendations"
    ]
    for item in toc_items:
        elements.append(Paragraph(item, body_style))
    elements.append(PageBreak())

    # Summary Section
    elements.append(Paragraph("<b>Summary</b>", subtitle_style))
    summary_text = f"""
    This report provides an in-depth analysis of customer churn trends. 
    The overall churn rate is {churn_rate:.1%} across {len(filtered_df):,} customers. 
    Key insights include churn distribution by category, predictive modeling results, 
    and actionable recommendations to reduce churn.
    """
    elements.append(Paragraph(summary_text, body_style))
    elements.append(PageBreak())

    # Overview Section
    elements.append(Paragraph("<b>Overview Metrics</b>", subtitle_style))
    elements.append(Paragraph(f"Overall Churn Rate: {churn_rate:.1%}", body_style))
    elements.append(Paragraph(f"Total Customers: {len(filtered_df):,}", body_style))
    elements.append(Spacer(1, 20))

    # Charts Section
    elements.append(Paragraph("<b>Visual Analysis</b>", subtitle_style))

    def add_chart(fig, caption):
        if fig:
            buf_img = io.BytesIO()
            fig.savefig(buf_img, format="PNG", dpi=150)
            buf_img.seek(0)
            elements.append(Image(buf_img, width=5*inch, height=3*inch))
            elements.append(Paragraph(caption, body_style))
            elements.append(Spacer(1, 20))

    add_chart(fig_cat, "Churn Rate by Category")
    add_chart(fig_roc, "ROC Curve")
    add_chart(fig_feat, "Top 10 Features Impacting Churn")

    # Recommendations Section
    elements.append(Paragraph("<b>Recommendations to Reduce Churn</b>", subtitle_style))
    for rec in recommendations:
        elements.append(Paragraph(f"✅ {rec}", body_style))

    # Build PDF with custom cover page and footer
    doc.build(elements, onFirstPage=lambda c, d: draw_cover_page(c, d, logo_path, bg_image_path), onLaterPages=add_page_number)
    return buf.getvalue()

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.title("📊 Customer Churn Dashboard")

    # Filters
    st.sidebar.subheader("Filters")
    country_filter = st.sidebar.multiselect("Select Country", options=df['country'].unique())
    gender_filter = st.sidebar.multiselect("Select Gender", options=df['gender'].unique())

    filtered_df = df.copy()
    if country_filter:
        filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]

    # KPI Cards
    churn_rate = filtered_df['churn'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Rate", f"{churn_rate:.1%}")
    col2.metric("Active Customers", f"{len(filtered_df):,}")
    col3.metric("MRR", "$50,000")  # Placeholder

    # Churn by Category
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

    # Prepare Data for Model
    numeric_cols = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
                    'avg_session_time', 'features_used', 'usage_growth_rate', 'last_login_days_ago',
                    'support_tickets', 'avg_resolution_time', 'csat_score', 'escalations',
                    'email_open_rate', 'marketing_click_rate', 'nps_score', 'payment_failures',
                    'referral_count', 'total_revenue', 'monthly_fee']

    selected_cols = [col for col in numeric_cols + categorical_cols if col in filtered_df.columns]
    X = filtered_df[selected_cols]
    y = filtered_df['churn']
    X = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], drop_first=True)

    # Train Model
    with st.spinner("Training model..."):
        model, scaler = train_model(X, y)
    st.success("✅ Model trained successfully.")

    # ROC Curve
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

    # Feature Importance
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

    # Recommendations
    st.subheader("Recommendations to Reduce Churn")
    for rec in recommendations:
        st.write(f"✅ {rec}")

    # PDF Download
    st.subheader("Download Full Magazine-Style Report")
    logo_path = "company_logo.png"  # Replace with your logo file path
    bg_image_path = "cover_bg.jpg"  # Replace with your background image path
    pdf_data = build_magazine_pdf(churn_rate, filtered_df, fig_cat, fig_roc, fig_feat, recommendations, logo_path, bg_image_path)

    st.download_button(
        label="📥 Download Magazine-Style PDF Report",
        data=pdf_data,
        file_name="churn_magazine_report.pdf",
        mime="application/pdf"
    )
