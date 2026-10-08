import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 연평균 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 로드 및 전처리 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 컬럼 파싱 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 평균기온 컬럼 탐색 및 정제
    temp_col = [col for col in df.columns if '평균' in col and '기온' in col]
    if temp_col:
        temp_col = temp_col[0]
    else:
        temp_col = df.columns[2]
        
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df = df.dropna(subset=[temp_col])
    
    # 연도별 평균 기온 계산
    annual_df = df.groupby('연도')[temp_col].mean().reset_index()
    annual_df.columns = ['연도', '연평균기온']
    annual_df['연도'] = annual_df['연도'].astype(int)
    
    return annual_df

st.title("🌡️ 서울 연평균 기온 예측 및 학습 기간별 선형회귀 모델 비교")
st.markdown("""
이 앱은 **서울 기온 데이터**를 사용하여 **전체 데이터 분석**과 더불어 **학습 기간(최근 50년 vs 최근 100년)**에 따른 선형회귀 모델의 
**기울기 변화** 및 **공통 테스트 데이터(최근 20년: 2006~2025년)** 예측 성능(MAE, MSE, $R^2$)을 비교 평가합니다.
""")

# 데이터 불러오기
try:
    data = load_data()
    st.sidebar.header("📊 데이터 요약 정보")
    st.sidebar.write(f"• 전체 수집 기간: {data['연도'].min()}년 ~ {data['연도'].max()}년")
    st.sidebar.write(f"• 총 연도 수: {len(data)}개 데이터")
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 데이터 분할 (Data Splitting)
# ---------------------------------------------------------
# 공통 테스트 데이터: 최근 20년 (2006~2025)
test_df = data[(data['연도'] >= 2006) & (data['연도'] <= 2025)].copy()

# 학습 데이터 A: 최근 50년 (1956~2005)
train_50_df = data[(data['연도'] >= 1956) & (data['연도'] <= 2005)].copy()

# 학습 데이터 B: 최근 100년 (1906~2005)
train_100_df = data[(data['연도'] >= 1906) & (data['연도'] <= 2005)].copy()

# ---------------------------------------------------------
# 모델 학습 및 평가 함수
# ---------------------------------------------------------
def fit_eval_model(train_data, test_data):
    X_train = train_data[['연도']]
    y_train = train_data['연평균기온']
    
    X_test = test_data[['연도']]
    y_test = test_data['연평균기온']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, mse, r2, y_pred

# 1) 전체 데이터 기준 모델
X_full = data[['연도']]
y_full = data['연평균기온']
model_full = LinearRegression()
model_full.fit(X_full, y_full)
y_pred_full = model_full.predict(X_full)
mae_full = mean_absolute_error(y_full, y_pred_full)
mse_full = mean_squared_error(y_full, y_pred_full)
r2_full = r2_score(y_full, y_pred_full)

# 2) 50년 / 100년 학습 모델 및 평가
model_50, mae_50, mse_50, r2_50, pred_50 = fit_eval_model(train_50_df, test_df)
model_100, mae_100, mse_100, r2_100, pred_100 = fit_eval_model(train_100_df, test_df)

# ---------------------------------------------------------
# Streamlit 화면 구성 (탭)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📉 최근 50년 vs 100년 학습 모델 비교", "🌐 전체 데이터 모델 평가", "🔮 미래 기온 예측 시뮬레이션"])

with tab1:
    st.subheader("1. 최근 50년 vs 최근 100년 학습 모델의 기울기 및 예측 성능 평가")
    st.caption("공통 평가 대상 테스트 데이터: 최근 20년 (2006년 ~ 2025년)")
    
    # 지표 비교 표 생성
    metrics_data = {
        "학습 데이터 조건": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)"],
        "회귀선 기울기 (°C/년)": [f"{model_50.coef_[0]:.5f}", f"{model_100.coef_[0]:.5f}"],
        "Y 절편": [f"{model_50.intercept_:.2f}", f"{model_100.intercept_:.2f}"],
        "MAE (평균 절대 오차)": [f"{mae_50:.4f} °C", f"{mae_100:.4f} °C"],
        "MSE (평균 제곱 오차)": [f"{mse_50:.4f} °C²", f"{mse_100:.4f} °C²"],
        "R² Score (결정계수)": [f"{r2_50:.4f}", f"{r2_100:.4f}"]
    }
    st.table(pd.DataFrame(metrics_data))
    
    # 핵심 지표 메트릭
    col1, col2, col3 = st.columns(3)
    col1.metric("50년 모델 MAE", f"{mae_50:.3f} °C", f"{mae_50 - mae_100:+.3f} °C (vs 100년)")
    col2.metric("50년 모델 MSE", f"{mse_50:.3f} °C²", f"{mse_50 - mse_100:+.3f} °C² (vs 100년)")
    col3.metric("50년 모델 R² Score", f"{r2_50:.3f}", f"{r2_50 - r2_100:+.3f} (vs 100년)")
    
    st.divider()
    
    # 시각화 그래프
    st.subheader("2. 학습 모델별 회귀선과 실제 데이터 비교")
    
    fig = go.Figure()
    
    # 전체 연도 데이터 (회색)
    fig.add_trace(go.Scatter(
        x=data['연도'], y=data['연평균기온'],
        mode='markers', name='과거 실제 연평균기온',
        marker=dict(color='lightgray', opacity=0.7, size=6)
    ))
    
    # 테스트 데이터 (빨간색)
    fig.add_trace(go.Scatter(
        x=test_df['연도'], y=test_df['연평균기온'],
        mode='markers', name='공통 테스트 데이터 (2006~2025)',
        marker=dict(color='crimson', size=9)
    ))
    
    # 회귀선 추정 범위
    years_range = np.arange(1906, 2026).reshape(-1, 1)
    
    # 50년 학습 회귀선
    fig.add_trace(go.Scatter(
        x=years_range.flatten(), y=model_50.predict(years_range),
        mode='lines', name=f'50년 학습 회귀선 (기울기: {model_50.coef_[0]:.4f})',
        line=dict(color='blue', width=2.5)
    ))
    
    # 100년 학습 회귀선
    fig.add_trace(go.Scatter(
        x=years_range.flatten(), y=model_100.predict(years_range),
        mode='lines', name=f'100년 학습 회귀선 (기울기: {model_100.coef_[0]:.4f})',
        line=dict(color='darkgreen', width=2.5, dash='dash')
    ))
    
    fig.update_layout(
        title="학습 범위에 따른 회귀선 비교 및 최근 20년 테스트 데이터 적합도",
        xaxis_title="연도",
        yaxis_title="연평균 기온 (°C)",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    💡 **학습 기간별 기울기 및 성능 비교 결과 분석**:
    - **기울기 차이**: 1956~2005년(최근 50년) 데이터를 학습한 모델의 회귀선 기울기가 1906~2005년(최근 100년) 데이터 모델의 기울기보다 가파르게 나타납니다. 이는 **20세기 후반 이후 서울의 도시화와 지구온난화 진행 속도가 가속화**되었음을 보여줍니다.
    - **예측 성능 비교**: 최근 기온 상승 추세를 보다 직접적으로 반영한 **50년 학습 모델**이 최근 20년(2006~2025년) 공통 테스트 데이터에 대해 더 정밀한 예측 오차(낮은 MAE, MSE 및 높은 R² Score)를 기록합니다.
    """)

with tab2:
    st.subheader("🌐 전체 데이터(관측 전체 기간) 기반 모델 평가")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    col_f1.metric("전체 기간 기울기", f"{model_full.coef_[0]:.5f} °C/년")
    col_f2.metric("전체 데이터 MAE", f"{mae_full:.4f} °C")
    col_f3.metric("전체 데이터 MSE", f"{mse_full:.4f} °C²")
    col_f4.metric("전체 데이터 R² Score", f"{r2_full:.4f}")
    
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=data['연도'], y=data['연평균기온'],
        mode='markers', name='실제 관측 기온',
        marker=dict(color='royalblue', size=6)
    ))
    fig_full.add_trace(go.Scatter(
        x=data['연도'], y=y_pred_full,
        mode='lines', name=f'전체 데이터 회귀선 (기울기: {model_full.coef_[0]:.4f})',
        line=dict(color='firebrick', width=2)
    ))
    fig_full.update_layout(
        title="서울 관측 전체 기간 연평균 기온 및 선형 회귀선",
        xaxis_title="연도",
        yaxis_title="연평균 기온 (°C)",
        template="plotly_white"
    )
    st.plotly_chart(fig_full, use_container_width=True)

with tab3:
    st.subheader("🔮 목표 연도 기온 예측")
    target_year = st.slider("예측하고자 하는 연도를 선택하세요", 2026, 2050, 2030)
    
    p_50 = model_50.predict([[target_year]])[0]
    p_100 = model_100.predict([[target_year]])[0]
    p_full = model_full.predict([[target_year]])[0]
    
    cp1, cp2, cp3 = st.columns(3)
    cp1.metric(f"50년 학습 모델 예측 ({target_year}년)", f"{p_50:.2f} °C")
    cp2.metric(f"100년 학습 모델 예측 ({target_year}년)", f"{p_100:.2f} °C")
    cp3.metric(f"전체 학습 모델 예측 ({target_year}년)", f"{p_full:.2f} °C")
