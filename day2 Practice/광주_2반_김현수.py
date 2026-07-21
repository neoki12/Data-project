"""
작성자   : 광주캠퍼스_2반_김현수
최초 작성일 : 2026-07-21
수정일   : 2026-07-21
목적     : [실습 3] Pandas EDA · Polars Lazy · DuckDB SQL 비교
          sales_100k.csv를 활용해 기초 EDA와 IQR 이상치 제거를 수행한다.

구성
==============================
1) Pandas EDA        : 기초 탐색(df.info/isnull) + IQR 방법으로 amount 컬럼 이상치 제거
2) Pandas groupby     : region·category별 total/mean/count named aggregation
3) Polars Lazy API    : 2)와 동일한 집계를 scan_csv -> filter -> group_by ->
                        agg -> sort -> collect 체인으로 재현
4) DuckDB SQL + 성능 비교 : 2)/3)과 동일한 집계를 SQL로 작성하고,
                            timeit(number 통일)으로 세 도구의 실행 시간 비교

변경 내역
==============================
2026-07-21
  1. sales_100k.csv 로딩 함수(load_sales) 작성 - 파일 없음 예외 처리.
  2. df.info() / isnull().sum()으로 기초 탐색(basic_eda) 작성.
  3. IQR(Q1-1.5*IQR ~ Q3+1.5*IQR) 방법으로 amount 이상치를 제거하는
     remove_outliers_iqr 작성. amount 결측치는 이상치와 구분해 별도로 집계.
  4. df.shape 출력 추가, 결측치 수/비율을 함께 보여주도록 basic_eda 보강.
  5. IQR 제거된 데이터로 region·category별 total/mean/count를
     named aggregation(aggregate_region_category)으로 계산, 총매출
     내림차순 정렬. region/category 결측은 그룹이 사라지지 않도록 '미상' 처리.
  6. Polars Lazy API로 2)와 동일한 집계를 재현
     (compute_iqr_bounds_polars + aggregate_region_category_polars).
     scan_csv로 지연 평가 계획을 세우고 collect()에서 한 번에 실행.
  7. DuckDB SQL로 동일 집계 재현(compute_iqr_bounds_duckdb +
     aggregate_region_category_duckdb). pandas 파이프라인을
     run_pandas_pipeline()으로 함수화해, 세 도구를 timeit(number=5로
     통일)으로 공정하게 비교하는 benchmark_three_tools() 추가.
  8. timeit 반복 횟수를 10회로 조정. pandas/DuckDB 결과 출력 코드가
     거의 동일하게 중복돼 있던 것을 print_aggregation_preview()로
     통합해 코드 간결성 개선.
  9. pandas/DuckDB 결과 표를 Polars(ASCII_MARKDOWN)와 동일한 마크다운
     스타일(to_markdown())로 통일. IQR 상하한 계산식이 3곳(pandas/
     Polars/DuckDB)에 반복돼 있던 것을 iqr_bounds()로 통합하고,
     benchmark_three_tools()의 반복되던 timeit 호출 3줄을 dict 순회로 축약.
"""

import timeit
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

TIMEIT_REPEAT = 10  # 세 도구 모두 동일하게 사용

DATA_FILE = Path(__file__).parent / "sales_100k.csv"


def load_sales(path: Path = DATA_FILE) -> pd.DataFrame:
    """sales_100k.csv를 읽어 DataFrame으로 반환한다. 파일이 없으면 예외를 발생시킨다."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}") from e


def basic_eda(df: pd.DataFrame) -> None:
    """df.shape, df.info(), 컬럼별 결측치 수/비율을 출력한다."""
    print(
        f"========== df.shape ========== \n{df.shape} (행 {df.shape[0]:,} / 열 {df.shape[1]})"
    )
    print("\n========== df.info() ==========")
    df.info()

    missing_count = df.isnull().sum()
    missing_ratio = (missing_count / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {"결측치 수": missing_count, "결측치 비율(%)": missing_ratio}
    )

    print("\n========== Column별 결측치 수/비율 ==========")
    print(missing_summary)


def iqr_bounds(q1: float, q3: float) -> tuple[float, float]:
    """Q1, Q3로부터 IQR 정상 범위 (Q1-1.5*IQR, Q3+1.5*IQR)를 계산한다.
    pandas/Polars/DuckDB 세 구현이 동일한 공식을 각자 반복하지 않도록
    한 곳에서만 정의한다."""
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """IQR(사분위 범위) 방법으로 column의 이상치를 제거한 DataFrame을 반환한다.
    column이 결측인 행은 이상치가 아니라 결측치이므로 미리 제외한 뒤 계산한다."""
    valid = df.dropna(subset=[column])
    lower, upper = iqr_bounds(
        valid[column].quantile(0.25), valid[column].quantile(0.75)
    )
    return valid[valid[column].between(lower, upper)]


def aggregate_region_category(df: pd.DataFrame) -> pd.DataFrame:
    """region·category별 총매출(total)/평균(mean)/건수(count)를
    named aggregation으로 계산하고, 총매출을 내림차순으로 정렬해 반환한다.
    region·category가 결측인 행은 groupby에서 그룹째로 사라지므로,
    먼저 '미상'으로 채워 결측 데이터도 하나의 그룹으로 남긴다."""
    labeled = df.assign(
        region=df["region"].fillna("미상"),
        category=df["category"].fillna("미상"),
    )
    return (
        labeled.groupby(["region", "category"])
        .agg(
            total=("amount", "sum"),
            mean=("amount", "mean"),
            count=("amount", "count"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )


def compute_iqr_bounds_polars(
    path: Path, column: str = "amount"
) -> tuple[float, float]:
    """Polars Lazy API로 column의 IQR 상하한(Q1-1.5*IQR, Q3+1.5*IQR)을 계산한다.
    결측은 미리 제외한 뒤 사분위수를 구한다."""
    try:
        stats = (
            pl.scan_csv(path)
            .filter(pl.col(column).is_not_null())
            .select(
                pl.col(column).quantile(0.25, interpolation="linear").alias("q1"),
                pl.col(column).quantile(0.75, interpolation="linear").alias("q3"),
            )
            .collect()
        )
    except pl.exceptions.PolarsError as e:
        raise ValueError(f"IQR 계산 중 오류가 발생했습니다: {e}") from e

    return iqr_bounds(stats["q1"][0], stats["q3"][0])


def aggregate_region_category_polars(path: Path) -> pl.DataFrame:
    """2) aggregate_region_category()와 동일한 집계를 Polars Lazy API로 재현한다.
    scan_csv -> filter(IQR 정상 범위) -> group_by -> agg -> sort -> collect 순서."""
    lower, upper = compute_iqr_bounds_polars(path)

    return (
        pl.scan_csv(path)
        .filter(pl.col("amount").is_between(lower, upper))
        .with_columns(
            pl.col("region").fill_null("미상"),
            pl.col("category").fill_null("미상"),
        )
        .group_by(["region", "category"])
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("amount").mean().alias("mean"),
            pl.col("amount").count().alias("count"),
        )
        .sort("total", descending=True)
        .collect()
    )


def run_pandas_pipeline(path: Path = DATA_FILE) -> pd.DataFrame:
    """1)/2)에서 만든 함수들로 로딩 -> IQR 이상치 제거 -> region·category
    집계까지 한 번에 실행한다 (timeit 비교용, 출력 없음)."""
    df = load_sales(path)
    cleaned = remove_outliers_iqr(df, "amount")
    return aggregate_region_category(cleaned)


def compute_iqr_bounds_duckdb(
    path: Path, column: str = "amount"
) -> tuple[float, float]:
    """DuckDB SQL로 column의 IQR 상하한(Q1-1.5*IQR, Q3+1.5*IQR)을 계산한다."""
    try:
        stats = duckdb.sql(
            f"""
            SELECT
                quantile_cont({column}, 0.25) AS q1,
                quantile_cont({column}, 0.75) AS q3
            FROM '{path}'
            WHERE {column} IS NOT NULL
            """
        ).df()
    except duckdb.Error as e:
        raise ValueError(f"IQR 계산 중 오류가 발생했습니다: {e}") from e

    return iqr_bounds(float(stats["q1"][0]), float(stats["q3"][0]))


def aggregate_region_category_duckdb(path: Path) -> pd.DataFrame:
    """2)/3)과 동일한 집계를 DuckDB SQL로 재현한다. region·category 결측은
    COALESCE로 '미상' 처리하고, amount는 IQR 정상 범위만 남긴다."""
    lower, upper = compute_iqr_bounds_duckdb(path)
    try:
        return duckdb.sql(
            f"""
            SELECT
                COALESCE(region, '미상') AS region,
                COALESCE(category, '미상') AS category,
                SUM(amount) AS total,
                AVG(amount) AS mean,
                COUNT(amount) AS count
            FROM '{path}'
            WHERE amount BETWEEN {lower} AND {upper}
            GROUP BY COALESCE(region, '미상'), COALESCE(category, '미상')
            ORDER BY total DESC
            """
        ).df()
    except duckdb.Error as e:
        raise ValueError(f"집계 중 오류가 발생했습니다: {e}") from e


def print_aggregation_preview(title: str, df: pd.DataFrame, n: int = 15) -> None:
    """pandas/DuckDB 집계 결과(둘 다 total/mean/count 컬럼을 가진 pandas
    DataFrame)를 동일한 포맷으로 상위 n개 출력한다. 중복 출력 코드를 통합.
    Polars 쪽 출력(ASCII_MARKDOWN)과 스타일을 맞추기 위해 to_markdown() 사용."""
    print(f"\n========== {title} (총 {len(df)}개 그룹, 총매출 내림차순) ==========")
    preview = df.head(n).copy()
    preview["total"] = preview["total"].map(lambda x: f"{x:,.0f}")
    preview["mean"] = preview["mean"].map(lambda x: f"{x:,.0f}")
    preview["count"] = preview["count"].map(lambda x: f"{x:,}")
    print(preview.to_markdown(index=False))


def benchmark_three_tools(
    path: Path = DATA_FILE, number: int = TIMEIT_REPEAT
) -> dict[str, float]:
    """pandas/Polars/DuckDB 파이프라인을 동일한 number로 timeit 측정해
    {도구명: 총 소요 시간(초)} dict를 반환한다."""
    pipelines = {
        "pandas": lambda: run_pandas_pipeline(path),
        "polars": lambda: aggregate_region_category_polars(path),
        "duckdb": lambda: aggregate_region_category_duckdb(path),
    }
    return {name: timeit.timeit(fn, number=number) for name, fn in pipelines.items()}


def main():
    """데이터 로딩 -> 기초 EDA -> IQR 이상치 제거 -> region·category 집계까지 실행한다."""
    try:
        df = load_sales()
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    basic_eda(df)

    total = len(df)
    missing_amount = df["amount"].isna().sum()
    cleaned = remove_outliers_iqr(df, "amount")
    after = len(cleaned)
    outliers = total - missing_amount - after

    print("\n========== IQR 이상치 제거 (컬럼: amount) ==========")
    print(f"전체 행 수        : {total:,}")
    print(f"amount 결측 행 수  : {missing_amount:,}")
    print(f"이상치로 제거된 행 : {outliers:,}")
    print(f"제거 후 행 수      : {after:,}")

    region_category_stats = aggregate_region_category(cleaned)
    print_aggregation_preview("[pandas] region x category 집계", region_category_stats)

    polars_stats = aggregate_region_category_polars(DATA_FILE)
    print(
        f"\n========== [Polars Lazy] region x category 집계 "
        f"(총 {polars_stats.height}개 그룹, 총매출 내림차순) =========="
    )
    polars_preview = polars_stats.head(15).with_columns(
        pl.col("total").round(0), pl.col("mean").round(0)
    )
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        fmt_float="full",
        thousands_separator=True,
        tbl_rows=15,
    ):
        print(polars_preview)

    duckdb_stats = aggregate_region_category_duckdb(DATA_FILE)
    print_aggregation_preview("[DuckDB SQL] region x category 집계", duckdb_stats)

    print(
        f"\n========== timeit 성능 비교 (number={TIMEIT_REPEAT}, 3개 도구 동일) =========="
    )
    times = benchmark_three_tools()
    for name, elapsed in sorted(times.items(), key=lambda item: item[1]):
        print(
            f"{name:8s}: 총 {elapsed:.3f}초  (1회 평균 {elapsed / TIMEIT_REPEAT * 1000:,.1f}ms)"
        )


if __name__ == "__main__":
    main()
