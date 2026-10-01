import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 선형 회귀 모델을 생성하고 온난화 추세 및 미래/과거의 기온을 예측합니다."
)


# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

    # 데이터 읽기 (CSV)
    df = pd.read_csv(url, encoding="utf-8")

    # '날짜' 컬럼을 datetime 형식으로 변환 및 연도 추출
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
    df_all = load_and_process_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 1. 전체 기간 회귀 계산
# -----------------------------------------------------------------------------
X_all = df_all["경과연수"].values
Y_all = df_all["평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, Y_all, 1)
corr_all = np.corrcoef(X_all, Y_all)[0, 1]

# 100년당 기온 변화량 (전체)
slope_100y_all = slope_all * 100

# -----------------------------------------------------------------------------
# 2. 최근 20년 회귀 계산
# -----------------------------------------------------------------------------
max_available_year = df_all["연도"].max()
cutoff_year = max_available_year - 19  # 최근 20년 (예: 2006~2025)

df_recent = df_all[df_all["연도"] >= cutoff_year].copy()
X_recent = df_recent["경과연수"].values
Y_recent = df_recent["평균기온"].values

slope_recent, intercept_recent = np.polyfit(X_recent, Y_recent, 1)
corr_recent = np.corrcoef(X_recent, Y_recent)[0, 1]

# 100년당 기온 변화량 (최근 20년)
slope_100y_recent = slope_recent * 100

# -----------------------------------------------------------------------------
# 정보 요약 및 기울기 비교 화면 표시
# -----------------------------------------------------------------------------
min_year = int(df_all["연도"].min())
max_year = int(df_all["연도"].max())
count_years = len(df_all)

st.markdown("### 📊 데이터 요약")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("분석 대상 연도 개수", f"{count_years} 개")
col_s2.metric("시작 연도", f"{min_year} 년")
col_s3.metric("끝 연도", f"{max_year} 년")
col_s4.metric("전체 상관계수 (r)", f"{corr_all:.4f}")

st.divider()

st.markdown("### ⚡ 100년당 기온 상승 속도 비교")
st.write("회귀 직선의 기울기를 '100년당 상승 기온(°C)'으로 환산한 수치입니다.")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown("#### 🌐 전체 기간 기준")
    st.caption(f"{min_year}년 ~ {max_year}년 데이터 사용")
    st.markdown(
        f"<h1 style='text-align: center; color: #1F77B4;'>+{slope_100y_all:.2f} °C <span style='font-size:20px;'>/ 100년</span></h1>",
        unsafe_allow_html=True,
    )

with comp_col2:
    st.markdown(f"#### 🔥 최근 20년 기준 ({cutoff_year}~{max_year})")
    st.caption(f"{cutoff_year}년 ~ {max_year}년 데이터 사용")

    # 전체 대비 가속도 비교 (delta 값 표현)
    diff = slope_100y_recent - slope_100y_all
    diff_text = f"(전체 대비 +{diff:.2f} °C 더 빠름)" if diff > 0 else ""

    st.markdown(
        f"<h1 style='text-align: center; color: #FF4B4B;'>+{slope_100y_recent:.2f} °C <span style='font-size:20px;'>/ 100년</span></h1>",
        unsafe_allow_html=True,
    )
    if diff_text:
        st.markdown(
            f"<p style='text-align: center; color: #FF4B4B; font-weight: bold;'>{diff_text}</p>",
            unsafe_allow_html=True,
        )

st.divider()

# -----------------------------------------------------------------------------
# 연도 선택 및 예측 기온
# -----------------------------------------------------------------------------
st.markdown("### 🔮 연도별 예상 기온 예측")
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 전체 기준 예측 및 최근 20년 기준 예측
pred_temp_all = slope_all * (selected_year - 1908) + intercept_all
pred_temp_recent = slope_recent * (selected_year - 1908) + intercept_recent

pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    st.markdown(f"**전체 추세선 기준 {selected_year}년 예상**")
    st.markdown(
        f"<h2 style='text-align: center; color: #1F77B4;'>{pred_temp_all:.2f} °C</h2>",
        unsafe_allow_html=True,
    )

with pred_col2:
    st.markdown(f"**최근 20년 추세선 기준 {selected_year}년 예상**")
    st.markdown(
        f"<h2 style='text-align: center; color: #FF4B4B;'>{pred_temp_recent:.2f} °C</h2>",
        unsafe_allow_html=True,
    )

st.divider()

# -----------------------------------------------------------------------------
# Plotly 시각화
# -----------------------------------------------------------------------------
st.markdown("### 📈 서울 연평균 기온 및 추세선 비교")

# 회귀선 표시용 연도 범위 (1900년 ~ 2100년)
line_years = np.arange(1900, 2101)
line_X = line_years - 1908

line_Y_all = slope_all * line_X + intercept_all
line_Y_recent = slope_recent * line_X + intercept_recent

fig = go.Figure()

# 1. 전체 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df_all["연도"],
        y=df_all["평균기온"],
        mode="markers",
        name="관측 평균기온",
        marker=dict(size=7, color="#A0A0A0", opacity=0.7),
    )
)

# 2. 최근 20년 데이터 강조 (붉은색 점)
fig.add_trace(
    go.Scatter(
        x=df_recent["연도"],
        y=df_recent["평균기온"],
        mode="markers",
        name=f"최근 20년 관측값 ({cutoff_year}~)",
        marker=dict(size=8, color="#d62728"),
    )
)

# 3. 전체 기간 회귀 직선 (파란색)
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_Y_all,
        mode="lines",
        name=f"전체 추세선 (+{slope_100y_all:.2f}℃/100년)",
        line=dict(color="#1f77b4", width=2, dash="dash"),
    )
)

# 4. 최근 20년 회귀 직선 (주황/빨간색 실선)
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_Y_recent,
        mode="lines",
        name=f"최근 20년 추세선 (+{slope_100y_recent:.2f}℃/100년)",
        line=dict(color="#ff7f0e", width=3),
    )
)

# 5. 선택한 연도의 예측 위치 강조
fig.add_trace(
    go.Scatter(
        x=[selected_year, selected_year],
        y=[pred_temp_all, pred_temp_recent],
        mode="markers",
        name=f"선택 연도({selected_year}년) 예측",
        marker=dict(size=12, color=["#1f77b4", "#ff7f0e"], symbol="star"),
    )
)

# 레이아웃 설정
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    xaxis=dict(dtick=20),
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)
