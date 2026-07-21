"""
작성자   : 광주캠퍼스_2반_김현수
최초 작성일 : 2026-07-21
수정일   : 2026-07-21
목적     : [실습 4] 시각화 4종 · 통계 검정 · sklearn Pipeline
          sales_100k.csv를 실습 3과 동일하게 IQR 이상치를 제거한 뒤,
          시각화·통계 검정·ML Pipeline·인터랙티브 차트까지 이어서 진행한다.

구성
==============================
1) EDA 시각화 4종 (2x2 서브플롯) : 히스토그램+KDE / 박스플롯 / 월별 라인 / 상관 히트맵
2) 통계 검정                     : 서울 vs 부산 평균 매출 t-test, region x category 카이제곱
3) sklearn Pipeline 구성 + 저장  : ColumnTransformer + Ridge, 훈련/평가/저장/재로딩
4) Plotly 인터랙티브 차트 저장   : region x category 총매출 막대 차트, HTML로 저장

변경 내역
==============================
2026-07-21
  1. load_sales/iqr_bounds/remove_outliers_iqr - 실습 3과 동일한 로직으로
     데이터를 로딩하고 IQR 이상치를 제거하는 전처리 재구성(prepare_cleaned_data).
  2. plt.subplots(2, 2)로 히스토그램+KDE, 박스플롯, 월별 총매출 라인,
     수치형 변수 상관 히트맵 4종을 한 figure에 그리는 plot_eda_grid 작성.
     GUI 없는 환경을 고려해 Agg 백엔드로 PNG 파일에 저장.
  3. plot_eda_grid 실행 결과 제목/축/지역명 등 한글 라벨이 전부 깨짐
     (DejaVu Sans 폰트에 한글 글리프 없음, Glyph missing 경고 다수 발생).
     matplotlib.font_manager.fontManager.ttflist로 시스템에 설치된
     한글 폰트를 조회해 AppleGothic을 확인하고,
     plt.rcParams["font.family"]="AppleGothic" +
     ["axes.unicode_minus"]=False로 설정해 해결. 처음 추가할 때 이
     rcParams 설정 줄이 seaborn import보다 앞줄에 끼어들어가 import가
     한 곳에 모이지 않는 문제가 있어, seaborn import를 그 위로 옮기고
     rcParams 설정은 모든 import가 끝난 뒤로 재배치.
  4. scipy.stats.ttest_ind로 서울 vs 부산 평균 매출 차이 검정
     (run_ttest_region_pair), chi2_contingency로 region x category
     독립성 검정(run_chi2_region_category) 작성. 두 검정 모두 p-value
     수치 출력에 더해 p<0.05 기준 유의미 여부를 한 줄로 해석해 출력.
  5. ColumnTransformer(StandardScaler + OneHotEncoder) + Ridge를 하나의
     Pipeline으로 묶는 build_amount_pipeline 작성. train_and_save_pipeline
     에서 train/test 분리 -> fit -> predict -> score(R^2/RMSE) -> joblib.dump
     까지 수행하고, reload_and_verify_pipeline에서 joblib.load 후 동일한
     score가 나오는지 재검증.
  6. region/category 결측을 '미상'으로 채우는 코드가 카이제곱 검정/Pipeline
     학습 두 곳에 중복돼 있던 것을 fill_missing_categoricals()로 통합.
  7. Plotly Express로 region x category 총매출 막대 차트를 만들고
     fig.write_html()로 HTML 파일 저장(save_region_category_bar_chart).
     104p/105p Checkpoint·감점 대상·평가 기준 전체 재점검 완료.
"""

import matplotlib

matplotlib.use("Agg")  # GUI 없는 환경에서도 안전하게 파일로 저장

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.rcParams["font.family"] = "AppleGothic"  # 한글 라벨 깨짐 방지 (macOS)
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스 기호 깨짐 방지

DATA_FILE = Path(__file__).parent / "sales_100k.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"

NUMERIC_FEATURES = ["quantity", "unit_price", "customer_age"]
CATEGORICAL_FEATURES = ["region", "category", "payment_method", "customer_gender"]
TARGET_COLUMN = "amount"


def load_sales(path: Path = DATA_FILE) -> pd.DataFrame:
    """sales_100k.csv를 읽어 DataFrame으로 반환한다. 파일이 없으면 예외를 발생시킨다."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}") from e


def iqr_bounds(q1: float, q3: float) -> tuple[float, float]:
    """Q1, Q3로부터 IQR 정상 범위(Q1-1.5*IQR, Q3+1.5*IQR)를 계산한다."""
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """IQR 방법으로 column의 이상치를 제거한 DataFrame을 반환한다.
    column이 결측인 행은 이상치가 아니라 결측치이므로 미리 제외한다."""
    valid = df.dropna(subset=[column])
    lower, upper = iqr_bounds(
        valid[column].quantile(0.25), valid[column].quantile(0.75)
    )
    return valid[valid[column].between(lower, upper)]


def fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """region/category 결측을 '미상'으로 채운 DataFrame을 반환한다.
    통계 검정/Pipeline 학습/Plotly 집계에서 결측 행이 조용히 제외되지
    않도록 한 곳에서만 정의해 재사용한다."""
    return df.assign(
        region=df["region"].fillna("미상"),
        category=df["category"].fillna("미상"),
    )


def prepare_cleaned_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """실습 3과 동일하게 로딩 -> amount 기준 IQR 이상치 제거까지 실행해
    이후 시각화/통계 검정/Pipeline 학습에 공통으로 쓸 데이터를 반환한다."""
    df = load_sales(path)
    return remove_outliers_iqr(df, "amount")


def plot_eda_grid(df: pd.DataFrame, out_path: Path) -> None:
    """2x2 서브플롯으로 히스토그램+KDE / 박스플롯 / 월별 총매출 라인 /
    상관 히트맵 4종을 한 figure에 그려 out_path에 저장한다."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) 히스토그램 + KDE
    sns.histplot(df["amount"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("amount 분포 (히스토그램 + KDE)")
    axes[0, 0].set_xlabel("amount")

    # 2) 박스플롯 (지역별 amount 분포)
    sns.boxplot(data=df, x="region", y="amount", ax=axes[0, 1])
    axes[0, 1].set_title("지역별 amount 분포 (박스플롯)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3) 월별 총매출 라인
    monthly = (
        df.assign(month=pd.to_datetime(df["order_date"]).dt.to_period("M").astype(str))
        .groupby("month")["amount"]
        .sum()
        .sort_index()
    )
    axes[1, 0].plot(monthly.index, monthly.values, marker="o", color="steelblue")
    axes[1, 0].set_title("월별 총매출 추이")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].set_ylabel("총매출")

    # 4) 수치형 변수 상관 히트맵
    corr = df[["quantity", "unit_price", "customer_age", "amount"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title("수치형 변수 상관관계")

    plt.tight_layout()
    try:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    except OSError as e:
        raise OSError(f"EDA 시각화 저장에 실패했습니다: {out_path}") from e
    finally:
        plt.close(fig)


def run_ttest_region_pair(
    df: pd.DataFrame, region_a: str = "서울", region_b: str = "부산"
) -> None:
    """region_a vs region_b의 평균 amount 차이를 t-test로 검정하고,
    t통계량/p-value와 함께 p<0.05 기준 유의미 여부를 해석해 출력한다."""
    group_a = df.loc[df["region"] == region_a, "amount"]
    group_b = df.loc[df["region"] == region_b, "amount"]

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    print(f"\n========== t-test: {region_a} vs {region_b} 평균 매출 ==========")
    print(f"{region_a} 평균: {group_a.mean():,.0f}원 (n={len(group_a):,})")
    print(f"{region_b} 평균: {group_b.mean():,.0f}원 (n={len(group_b):,})")
    print(f"t통계량: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(
            f"해석: p({p_value:.4f}) < 0.05 이므로 두 지역의 평균 매출 차이는 통계적으로 유의미하다."
        )
    else:
        print(
            f"해석: p({p_value:.4f}) >= 0.05 이므로 두 지역의 평균 매출 차이는 통계적으로 유의미하지 않다."
        )


def run_chi2_region_category(df: pd.DataFrame) -> None:
    """region x category 독립성을 카이제곱 검정으로 확인하고,
    카이제곱 통계량/p-value와 함께 p<0.05 기준 유의미 여부를 해석해 출력한다."""
    labeled = fill_missing_categoricals(df)
    contingency = pd.crosstab(labeled["region"], labeled["category"])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    print("\n========== 카이제곱 검정: region x category 독립성 ==========")
    print(f"카이제곱 통계량: {chi2:.4f}, 자유도: {dof}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(
            f"해석: p({p_value:.4f}) < 0.05 이므로 지역과 카테고리는 서로 독립이 아니다 (연관 있음)."
        )
    else:
        print(
            f"해석: p({p_value:.4f}) >= 0.05 이므로 지역과 카테고리는 서로 독립이라고 볼 수 있다 (연관 없음)."
        )


def build_amount_pipeline() -> Pipeline:
    """수치형(quantity/unit_price/customer_age)은 표준화, 범주형(region/category/
    payment_method/customer_gender)은 원핫인코딩한 뒤 Ridge 회귀로 amount를
    예측하는 전처리+모델 Pipeline을 구성한다."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])


def train_and_save_pipeline(
    df: pd.DataFrame, model_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Pipeline을 훈련하고 테스트셋으로 예측 및 평가한 뒤 joblib으로 저장한다. 
    재로딩 검증에 쓸 (X_test, y_test)를 반환한다."""
    features = fill_missing_categoricals(df)
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = features[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_amount_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = pipeline.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n========== sklearn Pipeline 훈련/평가 ==========")
    print(f"train {len(X_train):,}건 / test {len(X_test):,}건")
    print(f"R^2: {r2:.4f}")
    print(f"RMSE: {rmse:,.0f}원")

    try:
        joblib.dump(pipeline, model_path)
    except OSError as e:
        raise OSError(f"모델 저장에 실패했습니다: {model_path}") from e
    print(f"모델 저장 완료: {model_path}")

    return X_test, y_test


def reload_and_verify_pipeline(
    model_path: Path, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """저장된 Pipeline을 joblib으로 다시 읽어, 저장 직전과 동일한 R^2가
    나오는지 재로딩 검증한다."""
    try:
        loaded_pipeline = joblib.load(model_path)
    except (OSError, EOFError) as e:
        raise OSError(f"모델을 재로딩하지 못했습니다: {model_path}") from e

    reloaded_r2 = loaded_pipeline.score(X_test, y_test)
    print(f"재로딩한 모델의 R^2: {reloaded_r2:.4f} (저장 전과 동일해야 정상)")


def build_region_category_totals(df: pd.DataFrame) -> pd.DataFrame:
    """region x category별 총매출(amount 합계)을 계산한다.
    결측은 fill_missing_categoricals()로 '미상' 처리해 누락 없이 포함한다."""
    labeled = fill_missing_categoricals(df)
    return (
        labeled.groupby(["region", "category"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "total"})
    )


def save_region_category_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """region x category별 총매출을 Plotly Express 막대 차트로 만들어
    인터랙티브 HTML 파일로 저장한다 (화면 출력 대신 write_html 사용)."""
    totals = build_region_category_totals(df)
    fig = px.bar(
        totals,
        x="region",
        y="total",
        color="category",
        barmode="group",
        title="지역 x 카테고리별 총매출",
        labels={"total": "총매출", "region": "지역", "category": "카테고리"},
    )
    try:
        fig.write_html(out_path)
    except OSError as e:
        raise OSError(f"Plotly 차트 저장에 실패했습니다: {out_path}") from e


def main():
    """실습 3과 동일하게 데이터를 정제한 뒤 1) EDA 시각화, 2) 통계 검정,
    3) sklearn Pipeline 훈련/저장/재로딩, 4) Plotly 차트 저장까지 실행한다."""
    try:
        cleaned = prepare_cleaned_data()
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig_path = OUTPUT_DIR / "eda_grid.png"
    try:
        plot_eda_grid(cleaned, fig_path)
        print(f"1) EDA 시각화 4종 저장 완료: {fig_path}")
    except OSError as e:
        print(f"[오류] {e}")

    run_ttest_region_pair(cleaned)
    run_chi2_region_category(cleaned)

    model_path = OUTPUT_DIR / "amount_pipeline.joblib"
    try:
        X_test, y_test = train_and_save_pipeline(cleaned, model_path)
        reload_and_verify_pipeline(model_path, X_test, y_test)
    except OSError as e:
        print(f"[오류] {e}")

    chart_path = OUTPUT_DIR / "region_category_totals.html"
    try:
        save_region_category_bar_chart(cleaned, chart_path)
        print(f"\n4) Plotly 인터랙티브 차트 저장 완료: {chart_path}")
    except OSError as e:
        print(f"[오류] {e}")


if __name__ == "__main__":
    main()
