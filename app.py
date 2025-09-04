import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from utils import get_news_data, classify_sentiment

# Download vader_lexicon if not already present
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Portfolio Sentiment Analyzer", layout="wide")
st.title("📈 Portfolio News Sentiment Analyzer")

TICKERS = ['ACWI', 'IDEV', 'IEMG', 'SPY', 'QQQ', 'EWY', 'XLK', 'XLC', 'XLI', 'XLV', 'XLF', 'XLU', 'XLE', 'XLB', 'XLRE', 'XLY', 'XLP', 'SPYV', 'SPYG', 'VTV', 'VUG', 'VYM', 'RSP', 'USMV', 'PTF', 'SPMO']

st.sidebar.header("Settings")
tickers_selected = st.sidebar.multiselect("Select Tickers", TICKERS, default=TICKERS)
weeks_back = st.sidebar.slider("Weeks Back", 1, 8, 1)
show_raw = st.sidebar.checkbox("Show Raw Data")

# Date range calculation
today = datetime.now()
start_date = today - timedelta(weeks=weeks_back)

st.info(f"Analyzing sentiment from **{start_date.date()}** to **{today.date()}** for {len(tickers_selected)} tickers.")

with st.spinner("Fetching news and analyzing sentiment..."):
    df = get_news_data(tickers_selected, start_date)
    if not df.empty:
        sid = SentimentIntensityAnalyzer()
        df['Sentiment'] = df['Headline'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        df['Sentiment_Category'] = df['Sentiment'].apply(classify_sentiment)
        df = df.dropna(subset=['Date'])
    else:
        st.error("No news data found for the selected tickers and date range.")

if not df.empty:
    mean_values = df.groupby('Ticker')['Sentiment'].mean().reset_index()
    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    cm = sns.color_palette("twilight", 20)
    plot = sns.histplot(data=df['Sentiment'], kde=True, bins=20, ax=ax1)
    for bin_, i in zip(plot.patches, cm):
        bin_.set_facecolor(i)
    ax1.set_title('Sentiment Distribution')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

    st.subheader("Sentiment Score by Ticker")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax = sns.boxplot(x='Ticker', y='Sentiment', data=df, palette='twilight', ax=ax2)
    ax2.set_title('Sentiment Score')
    ax2.set_xlabel('Stock')
    ax2.set_ylabel('Sentiment Score')
    ax2.tick_params(axis='x', labelsize=10)
    for i, row in mean_values.iterrows():
        color = 'red' if row['Sentiment'] >= 0 else 'blue'
        ax2.text(i, row['Sentiment'], f'{row['Sentiment']:.2f}', ha='center', color=color, weight='semibold', fontsize=10)
    st.pyplot(fig2)

    st.subheader("Portfolio Sentiment Category Count")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    plot3 = sns.countplot(x='Sentiment_Category', data=df, palette='twilight', ax=ax3)
    ax3.set_title('Portfolio Sentiment')
    ax3.set_xlabel('Sentiment Category')
    ax3.set_ylabel('Number of Sources')
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    for p in ax3.patches:
        ax3.text(p.get_x() + p.get_width()/2, p.get_height()*0.7, f'{int(p.get_height())}', ha='center', va='center', fontsize=12, color='crimson', weight='bold')
    st.pyplot(fig3)

    st.subheader("Summary Table")
    st.dataframe(mean_values.style.background_gradient(cmap='twilight'), use_container_width=True)

    if show_raw:
        st.subheader("Raw News Data")
        st.dataframe(df, use_container_width=True)
else:
    st.warning("No data to display. Try a wider date range or different tickers.")