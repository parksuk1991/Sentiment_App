# Portfolio News Sentiment Analyzer

Streamlit 앱으로, 주요 ETF/주식 포트폴리오의 최근 뉴스 헤드라인을 수집하고 Sentiment(감성) 분석 및 시각화합니다.

## 기능
- Ticker별로 최근 n주간 뉴스 헤드라인 수집(yfinance)
- VADER를 이용한 감성점수 산출 및 Positive/Neutral/Negative 분류
- Sentiment 분포 히스토그램, 박스플롯, 카운트플롯 시각화
- Ticker 선택, 기간 조정, Raw 데이터 보기 옵션

## 사용법

1. **설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **실행**
   ```bash
   streamlit run app.py
   ```

3. **옵션 조정**
   - 사이드바에서 Ticker/기간/옵션 선택 가능

## 파일 설명

- `app.py` : Streamlit 메인 앱
- `utils.py` : 뉴스 데이터 수집 및 Sentiment 분류 함수
- `requirements.txt` : 필요한 패키지 목록
- `README.md` : 설명서

---

**네트워크 환경 및 yfinance API 변화에 따라 뉴스가 일부 ticker에서 제공되지 않을 수 있습니다.**