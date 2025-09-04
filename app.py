import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from utils import get_news_data, classify_sentiment

import requests
from PIL import Image
from io import BytesIO

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="포트폴리오 뉴스 감성 분석기", layout="wide")

# ===== 우측 상단 이미지 및 만든이 표시 =====
col_title, col_img_credit = st.columns([8, 1])
with col_title:
    st.title("📈 포트폴리오 센티먼트 분석기")
with col_img_credit:
    image_url = "https://amateurphotographer.com/wp-content/uploads/sites/7/2017/08/Screen-Shot-2017-08-23-at-22.29.18.png?w=600.jpg" # for parksuk1991
    fallback_icon = "https://cdn-icons-png.flaticon.com/512/3211/3211357.png"
    img_displayed = False
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        st.image(img, width=120, caption=None)
        img_displayed = True
    except Exception:
        try:
            response = requests.get(fallback_icon, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, width=120, caption=None)
            img_displayed = True
        except Exception:
            st.info("이미지를 불러올 수 없습니다.")

    st.markdown(
        "<div style='margin-top: -1px; text-align:center;'>"
        "<span style='font-size:1.05rem; color:#888;'>만든이: 박석</span>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align: center; margin-bottom: 6px;">'
        '<span style="font-size: 1.05rem;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</span></div>',
        unsafe_allow_html=True
    )
# ===== 여기까지 =====

TICKERS = ['ACWI', 'IDEV', 'IEMG', 'SPY', 'QQQ', 'EWY', 'XLK', 'XLC', 'XLI', 'XLV', 'XLF', 'XLU', 'XLE', 'XLB', 'XLRE', 'XLY', 'XLP', 'SPYV', 'SPYG', 'VTV', 'VUG', 'VYM', 'RSP', 'USMV', 'PTF', 'SPMO']

st.sidebar.header("설정")
tickers_selected = st.sidebar.multiselect("티커 선택", TICKERS, default=TICKERS)
weeks_back = st.sidebar.slider("최근 N 주 분석(1~12주)", 1, 12, 1)
show_raw = st.sidebar.checkbox("원본 데이터 보기")

today = datetime.now()
start_date = today - timedelta(weeks=weeks_back)

st.info(f"**{start_date.date()}** ~ **{today.date()}** 기간의 {len(tickers_selected)}개 티커 뉴스 감성 분석 결과입니다.")

with st.spinner("뉴스 수집 및 감성 분석 중..."):
    df = get_news_data(tickers_selected, start_date)
    if not df.empty:
        sid = SentimentIntensityAnalyzer()
        df['Sentiment'] = df['Headline'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        df['Sentiment_Category'] = df['Sentiment'].apply(classify_sentiment)
        df = df.dropna(subset=['Date'])
    else:
        st.error("선택한 티커와 기간에 대한 뉴스 데이터가 없습니다.")

if not df.empty:
    mean_values = df.groupby('Ticker')['Sentiment'].mean().reset_index()

    col1, col2, col3 = st.columns([1,1.5,1])

    with col1:
        st.markdown("#### 센티먼트 점수 분포")
        fig1, ax1 = plt.subplots(figsize=(5,2))
        cm = sns.color_palette("twilight", 20)
        plot = sns.histplot(data=df['Sentiment'], kde=True, bins=20, ax=ax1)
        for bin_, i in zip(plot.patches, cm):
            bin_.set_facecolor(i)
        ax1.set_title('Sentiment Score Distribution', fontsize=12)
        ax1.set_xlabel('Sentiment Score', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 자산별 센티먼트 점수")
        fig2, ax2 = plt.subplots(figsize=(7,2))
        ax = sns.boxplot(x='Ticker', y='Sentiment', data=df, palette='twilight', ax=ax2)
        ax2.set_title('Sentiment Score by Asset', fontsize=12)
        ax2.set_xlabel('Ticker', fontsize=10)
        ax2.set_ylabel('Sentiment Score', fontsize=10)
        ax2.tick_params(axis='x', labelsize=7)
        for i, row in mean_values.iterrows():
            color = 'red' if row['Sentiment'] >= 0 else 'blue'
            ax2.text(i, row['Sentiment'], f'{row["Sentiment"]:.2f}', ha='center', color=color, weight='semibold', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    with col3:
        st.markdown("#### 감성 카테고리 빈도")
        fig3, ax3 = plt.subplots(figsize=(4,2))
        plot3 = sns.countplot(x='Sentiment_Category', data=df, palette='twilight', ax=ax3)
        ax3.set_title('Sentiment Category Frequency', fontsize=12)
        ax3.set_xlabel('Sentiment Category', fontsize=10)
        ax3.set_ylabel('# of News', fontsize=10)
        ax3.tick_params(axis='x', labelsize=12)
        ax3.tick_params(axis='y', labelsize=8)
        for p in ax3.patches:
            ax3.text(p.get_x() + p.get_width()/2, p.get_height()*0.7, f'{int(p.get_height())}', ha='center', va='center', fontsize=10, color='crimson', weight='bold')
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("#### 자산별 평균 센티먼트 점수")
    st.dataframe(mean_values.style.background_gradient(cmap='twilight'), use_container_width=True, height=200)

    if show_raw:
        st.markdown("#### 원본 뉴스 데이터")
        st.dataframe(df, use_container_width=True, height=250)
else:
    st.warning("데이터가 없습니다. 기간을 늘리거나 티커 선택을 변경해 주세요.")
