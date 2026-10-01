import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 선형 회귀 모델을 생성하고 미래/과거의 기온을 예측합니다."
)


# 데이터 로드 함수
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

    # 데이터 읽기 (CSV)
    df = pd.read_csv(url, encoding="utf-8")

    # '날짜' 컬럼을 datetime 형식으로 변환 및 연도 extraction
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 1. 2025년 이하 데이터만 필터링
    df = df[df["연도"] <= 2025]

    # 2. 연도별 관측일수 및 평균기온 계산
    yearly_stats = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 3. 관측일수 300일 이상인 연도만 추출
    filtered_df = yearly_stats[yearly_stats["관측일수"] >= 300].copy()

    # 4. 회귀 분석을 위한 독립 변수 X (1908년부터 지난 연수)
    filtered_df["경과연수"] = filtered_df["연도"] - 1908

    return filtered_df


# 데이터 불러오기
try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 선형 회귀 계산 (numpy polyfit)
X = df["경과연수"].values
Y = df["평균기온"].values

slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr = np.corrcoef(X, Y)[0, 1]

# 정보 요약 표시
min_year = int(df["연도"].min())
max_year = int(df["연도"].max())
count_years = len(df)

st.markdown("### 📊 데이터 요약 및 분석 기준")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 개수", f"{count_years} 개")
col2.metric("시작 연도", f"{min_year} 년")
col3.metric("끝 연도", f"{max_year} 년")
col4.metric("상관계수 (r)", f"{corr:.4f}")

st.divider()

# 연도 선택 슬라이더
st.markdown("### 🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 예측 기온 계산 (1908년 기준 경과 연수 적용)
predicted_temp = slope * (selected_year - 1908) + intercept

# 예측 결과 강조 표시
st.subheader(f"📍 {selected_year}년 예상 평균기온")
st.markdown(
    f"<h1 style='text-align: center; color: #FF4B4B;'>{predicted_temp:.2f} °C</h1>",
    unsafe_allow_html=True,
)

st.divider()

# Plotly 그래프 생성
st.markdown("### 📈 서울 연평균 기온 및 회귀 직선")

# 회귀선 표시용 연도 범위 (1900년 ~ 2100년)
line_years = np.arange(1900, 2101)
line_X = line_years - 1908
line_Y = slope * line_X + intercept

fig = go.Figure()

# 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="관측 평균기온",
        marker=dict(size=8, color="#1f77b4"),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_Y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#ff7f0e", width=2),
    )
)

# 선택한 연도의 예측 지점 강조
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(size=14, color="red", symbol="star"),
    )
)

# 레이아웃 설정 (가로축 연도 그대로 표시)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    xaxis=dict(dtick=20),
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
