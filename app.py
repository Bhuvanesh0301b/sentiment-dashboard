# app.py
# Main Streamlit application

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agents.sentiment import load_sentiment_model, run_sentiment_analysis
from utils.preprocess import load_file
from config import SENTIMENT_COLORS

# ─────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit command
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS — makes the app look cleaner
# ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("📊 NLP Sentiment Analysis Dashboard")
st.markdown(
    "Upload product reviews to analyze customer sentiment using "
    "**BERT** (Bidirectional Encoder Representations from Transformers)"
)
st.divider()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    max_reviews = st.slider(
        label="Max reviews to analyze",
        min_value=50,
        max_value=5000,
        value=200,
        step=50,
        help="More reviews = slower but more complete analysis"
    )

    show_table = st.checkbox("Show full results table", value=True)
    show_samples = st.checkbox("Show example reviews per sentiment", value=True)

    st.divider()
    st.markdown("**About this app**")
    st.markdown(
        "Built with HuggingFace Transformers, "
        "Streamlit, and Plotly. "
        "Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`"
    )

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
with st.spinner("⏳ Loading BERT model (first load takes ~30 seconds)..."):
    model = load_sentiment_model()

st.success("✅ BERT model loaded and ready!")

# ─────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────
st.subheader("📁 Upload Your Reviews")

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        label="Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must have a column named 'review', 'text', or 'comment'"
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("📂 Use Sample Data", use_container_width=True)

# Load sample data if button clicked
if use_sample:
    uploaded_file = open("data/sample_reviews.csv", "rb")
    uploaded_file.name = "sample_reviews.csv"
    st.info("Using built-in sample dataset with 15 reviews.")

# ─────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Step 1: Load and preprocess file
    try:
        with st.spinner("Reading and cleaning your file..."):
            df, review_col = load_file(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Limit to max_reviews
    df = df.head(max_reviews)
    total_reviews = len(df)

    st.info(f"📋 Loaded **{total_reviews} reviews** from column: `{review_col}`")

    # Step 2: Run analysis
    with st.spinner(f"🤖 Analyzing {total_reviews} reviews with BERT... this may take a moment"):
        df = run_sentiment_analysis(df, review_col, model)

    st.success(f"✅ Analysis complete for {total_reviews} reviews!")
    st.divider()

    # ─────────────────────────────────────────────────────
    # KPI CARDS
    # ─────────────────────────────────────────────────────
    st.subheader("📈 Summary")

    pos_count = (df['Sentiment'] == 'Positive').sum()
    neg_count = (df['Sentiment'] == 'Negative').sum()
    neu_count = (df['Sentiment'] == 'Neutral').sum()
    avg_conf  = df['Confidence %'].mean()

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Reviews",    total_reviews)
    k2.metric("😊 Positive",      f"{pos_count}",
              delta=f"{pos_count*100//total_reviews}%")
    k3.metric("😡 Negative",      f"{neg_count}",
              delta=f"-{neg_count*100//total_reviews}%",
              delta_color="inverse")
    k4.metric("😐 Neutral",       f"{neu_count}",
              delta=f"{neu_count*100//total_reviews}%",
              delta_color="off")
    k5.metric("Avg Confidence",   f"{avg_conf:.1f}%")

    st.divider()

    # ─────────────────────────────────────────────────────
    # CHARTS — Row 1
    # ─────────────────────────────────────────────────────
    st.subheader("📊 Visual Analysis")

    chart1, chart2 = st.columns(2)

    # Pie chart
    with chart1:
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        fig_pie = px.pie(
            sentiment_counts,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment Distribution",
            hole=0.4    # donut chart
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar chart
    with chart2:
        fig_bar = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            title="Review Count by Sentiment",
            text='Count'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # CHARTS — Row 2
    # ─────────────────────────────────────────────────────
    chart3, chart4 = st.columns(2)

    # Confidence box plot
    with chart3:
        fig_box = px.box(
            df,
            x='Sentiment',
            y='Confidence %',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            title="Model Confidence per Sentiment",
            points="outliers"
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Confidence histogram
    with chart4:
        fig_hist = px.histogram(
            df,
            x='Confidence %',
            color='Sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            title="Confidence Score Distribution",
            nbins=20,
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────
    # SAMPLE REVIEWS PER SENTIMENT
    # ─────────────────────────────────────────────────────
    if show_samples:
        st.subheader("🔍 Sample Reviews per Sentiment")

        tab1, tab2, tab3 = st.tabs(["😊 Positive", "😡 Negative", "😐 Neutral"])

        with tab1:
            pos_samples = df[df['Sentiment'] == 'Positive'][
                [review_col, 'Confidence %']
            ].head(5)
            if len(pos_samples) > 0:
                st.dataframe(pos_samples, use_container_width=True)
            else:
                st.info("No positive reviews found.")

        with tab2:
            neg_samples = df[df['Sentiment'] == 'Negative'][
                [review_col, 'Confidence %']
            ].head(5)
            if len(neg_samples) > 0:
                st.dataframe(neg_samples, use_container_width=True)
            else:
                st.info("No negative reviews found.")

        with tab3:
            neu_samples = df[df['Sentiment'] == 'Neutral'][
                [review_col, 'Confidence %']
            ].head(5)
            if len(neu_samples) > 0:
                st.dataframe(neu_samples, use_container_width=True)
            else:
                st.info("No neutral reviews found.")

        st.divider()

    # ─────────────────────────────────────────────────────
    # FULL RESULTS TABLE
    # ─────────────────────────────────────────────────────
    if show_table:
        st.subheader("📄 Full Results Table")

        # Filter by sentiment
        filter_options = ["All", "Positive", "Negative", "Neutral"]
        selected_filter = st.selectbox("Filter by sentiment", filter_options)

        display_df = df.copy()
        if selected_filter != "All":
            display_df = display_df[display_df['Sentiment'] == selected_filter]

        st.dataframe(
            display_df[[review_col, 'Sentiment', 'Confidence %']].reset_index(drop=True),
            use_container_width=True,
            height=350
        )

    st.divider()

    # ─────────────────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────────────────
    st.subheader("⬇️ Download Results")

    result_csv = df[[review_col, 'Sentiment', 'Confidence %']].to_csv(index=False)

    st.download_button(
        label="Download Results as CSV",
        data=result_csv,
        file_name="sentiment_results.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    # Show instructions when no file is uploaded
    st.markdown("""
    ### How to use this app:
    1. **Upload** a CSV or Excel file with a column of reviews
    2. Or click **Use Sample Data** to try with example reviews
    3. Wait for BERT to analyze all reviews
    4. Explore the **charts and insights**
    5. **Download** the results CSV

    ### CSV format expected:
    Your file should have a column named one of these:
    `review`, `text`, `comment`, `feedback`, `description`

    ### Example:
    | review |
    |--------|
    | Great product, loved it! |
    | Terrible quality, broke in a day. |
    | Average, nothing special. |
    """)