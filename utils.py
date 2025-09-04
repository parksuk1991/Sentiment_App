import yfinance as yf
import pandas as pd
from datetime import datetime

def get_news_data(tickers, start_date):
    news_list = []
    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        try:
            news = ticker.news
        except Exception:
            news = []
        for article in news:
            content = article.get('content', {})
            date_str = content.get('pubDate')
            try:
                pubdate = pd.to_datetime(date_str)
            except Exception:
                pubdate = None
            if pubdate and pubdate >= start_date:
                news_list.append({
                    'Ticker': ticker_symbol,
                    'Date': pubdate,
                    'Headline': content.get('title', '')
                })
    df = pd.DataFrame(news_list)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'