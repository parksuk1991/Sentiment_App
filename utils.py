import yfinance as yf
import pandas as pd
from datetime import datetime

def get_news_data(tickers, start_date):
    news_list = []
    # start_date를 timezone-naive로 변환
    start_date_naive = start_date.replace(tzinfo=None)
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
                # pubdate를 timezone-naive로 변환
                if pubdate is not pd.NaT and pubdate is not None:
                    pubdate_naive = pubdate.replace(tzinfo=None)
                else:
                    pubdate_naive = None
            except Exception:
                pubdate_naive = None
            # pubdate_naive와 start_date_naive 비교
            if pubdate_naive and pubdate_naive >= start_date_naive:
                news_list.append({
                    'Ticker': ticker_symbol,
                    'Date': pubdate_naive,
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
