import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config for a professional look
st.set_page_config(page_title="AAPL 2020 Dashboard", layout="wide")

# Google Drive direct download URLs (replace with your actual URLs)
AAPL_DATA_URL = "https://drive.google.com/uc?export=download&id=14cW3TnFy2hkFHtKWDT6YdeaJzxql-HOz"
SENTIMENT_RETURNS_URL = "https://drive.google.com/uc?export=download&id=1Q19rAeP1JYdkSvFKrlH2ZaGWfeevU36B"

# Load data from Google Drive
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load datasets
aapl_df = load_data(AAPL_DATA_URL)
sentiment_df = load_data(SENTIMENT_RETURNS_URL)

# Check if data loaded successfully
if aapl_df is None or sentiment_df is None:
    st.stop()

# Dashboard Title and Introduction
st.title("AAPL 2020 Performance Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of AAPL's stock performance in 2020, including technical indicators and news sentiment correlation.  
Built for the 10 Academy AIM Week 1 Challenge.
""")

# Section 1: Key Metrics
st.header("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Year-End Close", f"${aapl_df['Close'].iloc[-1]:.2f}")
with col2:
    st.metric("Cumulative Return", f"{aapl_df['Cum_Returns'].iloc[-1] * 100:.1f}%")
with col3:
    st.metric("Max Volatility", f"{aapl_df['Volatility'].max():.3f}")

# Section 2: Stock Price and Technical Indicators
st.header("Stock Price and Technical Indicators")
fig_price = px.line(aapl_df, x='Date', y=['Close', 'SMA_20'], title="AAPL Closing Price and SMA (20-day)")
fig_price.update_layout(yaxis_title="Price ($)", legend_title="Metric")
st.plotly_chart(fig_price, use_container_width=True)

fig_rsi = px.line(aapl_df, x='Date', y='RSI_14', title="Relative Strength Index (RSI)")
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
fig_rsi.update_layout(yaxis_title="RSI")
st.plotly_chart(fig_rsi, use_container_width=True)

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=aapl_df['Date'], y=aapl_df['MACD'], name="MACD", line=dict(color="orange")))
fig_macd.add_trace(go.Scatter(x=aapl_df['Date'], y=aapl_df['MACD_Signal'], name="Signal", line=dict(color="blue")))
fig_macd.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="MACD", legend_title="Indicator")
st.plotly_chart(fig_macd, use_container_width=True)

# Section 3: Sentiment vs. Returns Correlation
st.header("News Sentiment vs. Stock Returns")
fig_sentiment = px.scatter(sentiment_df, x='Sentiment', y='Returns', trendline="ols",
                           title="Sentiment vs. AAPL Returns",
                           labels={"Sentiment": "Sentiment Score", "Returns": "Daily Returns (%)"})
st.plotly_chart(fig_sentiment, use_container_width=True)

# Correlation Metrics
st.subheader("Correlation Analysis")
col1, col2 = st.columns(2)
with col1:
    r_direct = sentiment_df['Sentiment'].corr(sentiment_df['Returns'])
    st.metric("Direct Correlation (lag=0)", f"{r_direct:.4f}")
with col2:
    # Lagged correlation (lag=1)
    lagged_df = sentiment_df.copy()
    lagged_df['Sentiment'] = lagged_df['Sentiment'].shift(-1)
    r_lag1 = lagged_df['Sentiment'].corr(lagged_df['Returns'])
    st.metric("Lagged Correlation (lag=1)", f"{r_lag1:.4f}")

# Section 4: Conclusion
st.header("Conclusion")
st.markdown("""
AAPL showed resilience in 2020, achieving a cumulative return of 58.3% despite early volatility from the COVID-19 crash.  
Technical indicators like RSI and MACD highlighted key trading signals, while sentiment analysis revealed a weak but insightful correlation with returns.
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit by Yitbie for 10 Academy AIM Week 1 Challenge")