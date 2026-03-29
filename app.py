"""
app.py
======
Main Streamlit web application for the Financial Market Crash Prediction System.

This file creates the entire web interface with 5 pages:
  1. Home            - Project overview and feature highlights
  2. Stock Dashboard - Interactive stock charts with moving averages
  3. Sentiment       - News headline sentiment analysis
  4. Crash Prediction- ML-based crash probability prediction
  5. Data Upload     - Upload and analyze custom CSV stock data

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import our custom modules
from data_fetch import fetch_stock_data, calculate_moving_averages, calculate_features
from sentiment import analyze_headlines, get_sample_headlines, get_average_sentiment
from model import train_model, predict_crash


# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Crash Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS FOR PREMIUM LOOK
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
    }
    .feature-card h3 {
        color: #818cf8;
        margin-bottom: 8px;
    }
    .feature-card p {
        color: #94a3b8;
        font-size: 14px;
    }

    /* Metric cards */
    .risk-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .risk-low { border-color: #22c55e; }
    .risk-medium { border-color: #eab308; }
    .risk-high { border-color: #f97316; }
    .risk-critical { border-color: #ef4444; }

    /* Hero section */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 8px;
    }

    /* Section headers */
    .section-header {
        color: #818cf8;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(129, 140, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
st.sidebar.markdown("## 📉 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Stock Dashboard", "📰 Sentiment Analysis", "🔮 Crash Prediction", "📁 Data Upload"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This app uses machine learning and sentiment analysis "
    "to predict potential financial market crashes."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center; color:#64748b; font-size:12px;'>"
    "Built with ❤️ using Streamlit & Python</p>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
# PAGE 1 : HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    # Hero
    st.markdown('<h1 class="hero-title">Financial Market Crash Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'AI-powered crash prediction using sentiment analysis &amp; historical market data'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Feature cards in 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<h3>📊 Stock Dashboard</h3>'
            '<p>Interactive charts with price history, volume, and moving averages for any stock ticker.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<h3>📰 Sentiment Analysis</h3>'
            '<p>Analyze financial news headlines using VADER to gauge market mood.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="feature-card">'
            '<h3>🔮 Crash Prediction</h3>'
            '<p>Machine learning model predicts crash probability with risk level assessment.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="feature-card">'
            '<h3>📁 Data Upload</h3>'
            '<p>Upload your own CSV stock data for custom analysis and visualization.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # How it works section
    st.markdown('<div class="section-header">🧠 How It Works</div>', unsafe_allow_html=True)

    st.markdown("""
    1. **Fetch Data** — Historical stock prices are downloaded via Yahoo Finance (yfinance).
    2. **Analyze Sentiment** — Financial news headlines are scored using VADER sentiment analysis.
    3. **Extract Features** — Price change, volume change, and volatility are computed from the data.
    4. **Predict** — A Random Forest model combines all features to estimate crash probability.
    """)

    # Tech stack
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)

    tech_cols = st.columns(4)
    tech_items = [
        ("🐍 Python", "Core language"),
        ("🎈 Streamlit", "Web framework"),
        ("📈 Plotly", "Interactive charts"),
        ("🤖 Scikit-learn", "ML models"),
    ]
    for col, (name, desc) in zip(tech_cols, tech_items):
        with col:
            st.metric(label=name, value=desc)


# ══════════════════════════════════════════════
# PAGE 2 : STOCK DASHBOARD
# ══════════════════════════════════════════════
elif page == "📊 Stock Dashboard":
    st.markdown('<h1 class="hero-title">Stock Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore historical stock data with interactive charts</p>', unsafe_allow_html=True)
    st.markdown("")

    # Input controls
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        ticker = st.text_input("🔍 Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, TSLA, RELIANCE.NS")
    with col_input2:
        period = st.selectbox("📅 Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    if st.button("📥 Fetch Data", use_container_width=True):
        with st.spinner(f"Fetching data for {ticker}..."):
            df = fetch_stock_data(ticker, period)

        if df.empty:
            st.error(f"❌ Could not fetch data for **{ticker}**. Please check the ticker symbol.")
        else:
            st.success(f"✅ Loaded {len(df)} days of data for **{ticker}**")

            # Add moving averages
            df = calculate_moving_averages(df)

            # ─── Price Chart ───
            st.markdown('<div class="section-header">💰 Price Chart with Moving Averages</div>', unsafe_allow_html=True)

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=df.index, y=df["Close"],
                mode="lines", name="Close Price",
                line=dict(color="#818cf8", width=2),
            ))
            fig_price.add_trace(go.Scatter(
                x=df.index, y=df["SMA_20"],
                mode="lines", name="SMA 20",
                line=dict(color="#22c55e", width=1.5, dash="dash"),
            ))
            fig_price.add_trace(go.Scatter(
                x=df.index, y=df["SMA_50"],
                mode="lines", name="SMA 50",
                line=dict(color="#f97316", width=1.5, dash="dot"),
            ))
            fig_price.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # ─── Volume Chart ───
            st.markdown('<div class="section-header">📊 Volume Chart</div>', unsafe_allow_html=True)

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker_color="#818cf8",
                opacity=0.7,
            ))
            fig_vol.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # ─── Key Metrics ───
            st.markdown('<div class="section-header">📋 Key Metrics</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            m2.metric("52-Day High", f"${df['High'].max():.2f}")
            m3.metric("52-Day Low", f"${df['Low'].min():.2f}")

            price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
            m4.metric("Period Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")

            # ─── Raw Data ───
            with st.expander("📄 View Raw Data"):
                st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 : SENTIMENT ANALYSIS
# ══════════════════════════════════════════════
elif page == "📰 Sentiment Analysis":
    st.markdown('<h1 class="hero-title">Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Analyze financial news headlines using VADER sentiment</p>', unsafe_allow_html=True)
    st.markdown("")

    # Option: use sample headlines or enter custom ones
    input_mode = st.radio("Choose input mode:", ["📰 Sample Headlines", "✍️ Enter Custom Headlines"], horizontal=True)

    if input_mode == "📰 Sample Headlines":
        headlines = get_sample_headlines()
        st.info(f"Using **{len(headlines)}** sample financial news headlines.")
    else:
        custom_text = st.text_area(
            "Enter headlines (one per line):",
            height=200,
            placeholder="Stock market rallies to record highs...\nMajor bank reports unexpected losses...",
        )
        headlines = [h.strip() for h in custom_text.strip().split("\n") if h.strip()]

    if st.button("🔍 Analyze Sentiment", use_container_width=True) and headlines:
        with st.spinner("Analyzing sentiment..."):
            results_df = analyze_headlines(headlines)
            avg_sentiment = get_average_sentiment(headlines)

        # ─── Overall Sentiment Metrics ───
        st.markdown('<div class="section-header">📊 Overall Sentiment</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Positive", f"{results_df['Positive'].mean():.3f}")
        s2.metric("Avg Negative", f"{results_df['Negative'].mean():.3f}")
        s3.metric("Avg Neutral", f"{results_df['Neutral'].mean():.3f}")
        s4.metric("Avg Compound", f"{avg_sentiment:.3f}",
                  delta="Positive" if avg_sentiment > 0 else "Negative")

        # ─── Sentiment Distribution Chart ───
        st.markdown('<div class="section-header">📈 Sentiment Distribution</div>', unsafe_allow_html=True)

        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Bar(
            x=list(range(len(results_df))),
            y=results_df["Compound"],
            marker_color=[
                "#22c55e" if v > 0.05 else "#ef4444" if v < -0.05 else "#94a3b8"
                for v in results_df["Compound"]
            ],
            text=[f"{v:.2f}" for v in results_df["Compound"]],
            textposition="outside",
        ))
        fig_sentiment.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Headline Index",
            yaxis_title="Compound Score",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # ─── Pie Chart: Positive vs Negative vs Neutral ───
        positive_count = (results_df["Compound"] > 0.05).sum()
        negative_count = (results_df["Compound"] < -0.05).sum()
        neutral_count = len(results_df) - positive_count - negative_count

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[positive_count, negative_count, neutral_count],
            marker_colors=["#22c55e", "#ef4444", "#94a3b8"],
            hole=0.4,
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
        )

        pie_col1, pie_col2 = st.columns([1, 1])
        with pie_col1:
            st.markdown('<div class="section-header">🎯 Sentiment Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        with pie_col2:
            st.markdown('<div class="section-header">📄 Detailed Results</div>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, height=350)


# ══════════════════════════════════════════════
# PAGE 4 : CRASH PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮 Crash Prediction":
    st.markdown('<h1 class="hero-title">Crash Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">ML-powered market crash probability estimation</p>', unsafe_allow_html=True)
    st.markdown("")

    # Train the model (cached so it only runs once)
    @st.cache_resource
    def get_trained_model():
        """Train and cache the model so we don't retrain on every interaction."""
        return train_model()

    model, accuracy = get_trained_model()
    st.success(f"✅ Model trained successfully — Test accuracy: **{accuracy * 100:.1f}%**")

    st.markdown("---")

    # Input: stock ticker + sentiment from headlines
    col_pred1, col_pred2 = st.columns([2, 1])
    with col_pred1:
        pred_ticker = st.text_input("🔍 Stock Ticker for Prediction", value="AAPL", key="pred_ticker")
    with col_pred2:
        pred_period = st.selectbox("📅 Period", ["1mo", "3mo", "6mo", "1y"], index=3, key="pred_period")

    if st.button("🔮 Predict Crash Probability", use_container_width=True):
        with st.spinner("Fetching data and running prediction..."):
            # Step 1: Fetch stock data
            df = fetch_stock_data(pred_ticker, pred_period)

            if df.empty:
                st.error(f"❌ Could not fetch data for **{pred_ticker}**.")
            else:
                # Step 2: Calculate stock features
                features = calculate_features(df)

                if features is None:
                    st.error("❌ Not enough data to calculate features (need at least 20 days).")
                else:
                    # Step 3: Get sentiment score from sample headlines
                    headlines = get_sample_headlines()
                    sentiment_score = get_average_sentiment(headlines)
                    features["sentiment_score"] = sentiment_score

                    # Step 4: Predict
                    result = predict_crash(model, features)

                    # ─── Display Results ───
                    st.markdown("---")
                    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

                    # Big metrics
                    r1, r2, r3 = st.columns(3)

                    # Color based on risk level
                    risk_colors = {
                        "LOW": "#22c55e",
                        "MEDIUM": "#eab308",
                        "HIGH": "#f97316",
                        "CRITICAL": "#ef4444",
                    }
                    risk_color = risk_colors.get(result["risk_level"], "#94a3b8")

                    with r1:
                        st.markdown(
                            f'<div class="risk-card">'
                            f'<h2 style="color:{risk_color}; font-size:3rem; margin:0;">'
                            f'{result["crash_probability"]:.1f}%</h2>'
                            f'<p style="color:#94a3b8; margin-top:8px;">Crash Probability</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with r2:
                        st.markdown(
                            f'<div class="risk-card risk-{result["risk_level"].lower()}">'
                            f'<h2 style="color:{risk_color}; font-size:3rem; margin:0;">'
                            f'{result["risk_level"]}</h2>'
                            f'<p style="color:#94a3b8; margin-top:8px;">Risk Level</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with r3:
                        pred_color = "#ef4444" if result["prediction"] == "Crash Likely" else "#22c55e"
                        st.markdown(
                            f'<div class="risk-card">'
                            f'<h2 style="color:{pred_color}; font-size:2rem; margin:0;">'
                            f'{result["prediction"]}</h2>'
                            f'<p style="color:#94a3b8; margin-top:8px;">Prediction</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # ─── Crash Probability Gauge ───
                    st.markdown("")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["crash_probability"],
                        title={"text": "Crash Probability", "font": {"color": "white"}},
                        number={"suffix": "%", "font": {"color": "white"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "white"},
                            "bar": {"color": risk_color},
                            "bgcolor": "rgba(255,255,255,0.1)",
                            "steps": [
                                {"range": [0, 25], "color": "rgba(34,197,94,0.3)"},
                                {"range": [25, 50], "color": "rgba(234,179,8,0.3)"},
                                {"range": [50, 75], "color": "rgba(249,115,22,0.3)"},
                                {"range": [75, 100], "color": "rgba(239,68,68,0.3)"},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "white"},
                        height=300,
                        margin=dict(l=30, r=30, t=50, b=0),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # ─── Feature Breakdown ───
                    st.markdown('<div class="section-header">📋 Feature Breakdown</div>', unsafe_allow_html=True)

                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Price Change", f"{features['price_change']:.2f}%")
                    f2.metric("Volume Change", f"{features['volume_change']:.2f}%")
                    f3.metric("Volatility", f"{features['volatility']:.2f}%")
                    f4.metric("Sentiment Score", f"{features['sentiment_score']:.3f}")


# ══════════════════════════════════════════════
# PAGE 5 : DATA UPLOAD
# ══════════════════════════════════════════════
elif page == "📁 Data Upload":
    st.markdown('<h1 class="hero-title">Data Upload</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload your own CSV stock data for analysis</p>', unsafe_allow_html=True)
    st.markdown("")

    st.info(
        "📄 **Expected CSV format:** Columns should include `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.\n\n"
        "You can use the `sample_stock_data.csv` included in the project as a reference."
    )

    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Uploaded **{uploaded_file.name}** — {len(df)} rows, {len(df.columns)} columns")

        # ─── Data Preview ───
        st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        # ─── Basic Statistics ───
        st.markdown('<div class="section-header">📊 Basic Statistics</div>', unsafe_allow_html=True)

        if "Close" in df.columns:
            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Mean Price", f"${df['Close'].mean():.2f}")
            stat2.metric("Max Price", f"${df['Close'].max():.2f}")
            stat3.metric("Min Price", f"${df['Close'].min():.2f}")
            stat4.metric("Std Dev", f"${df['Close'].std():.2f}")

        st.dataframe(df.describe(), use_container_width=True)

        # ─── Price Chart (if Close column exists) ───
        if "Close" in df.columns:
            st.markdown('<div class="section-header">📈 Price Chart</div>', unsafe_allow_html=True)

            # Try to use Date column for x-axis
            x_axis = df["Date"] if "Date" in df.columns else df.index

            fig_upload = go.Figure()
            fig_upload.add_trace(go.Scatter(
                x=x_axis, y=df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#818cf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(129, 140, 248, 0.1)",
            ))

            # Add volume as secondary y-axis if available
            if "Volume" in df.columns:
                fig_upload.add_trace(go.Bar(
                    x=x_axis, y=df["Volume"],
                    name="Volume",
                    marker_color="rgba(129, 140, 248, 0.3)",
                    yaxis="y2",
                ))
                fig_upload.update_layout(
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )

            fig_upload.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=450,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_upload, use_container_width=True)

        # ─── Download processed data ───
        st.markdown("---")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Processed Data",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
