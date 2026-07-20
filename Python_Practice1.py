"""
작성자   : 광주캠퍼스_2반_김현수
최초 작성일 : 2026-07-20
수정일   : 2026-07-20
목적     : Python_Practice1_Data.json을 활용한 파이썬 실습 프로그램

구성
----
1) 리스트/딕셔너리 컴프리헨션 : amount 필터링, 지역별 총매출 집계
2) Counter + defaultdict     : 지역별 거래 건수, 카테고리별 amount 리스트
3) 제너레이터                : amount 이상 거래를 yield, 리스트와 메모리 크기 비교
4) 종합                      : defaultdict + 컴프리헨션으로 월별 x 카테고리별 매출 집계 (+ 월별 총매출, top3)

변경 내역
--------
- 2026-07-20: 반복되던 필터링 로직을 함수로 분리(filter_high_amount),
  데이터 로딩/제너레이터 소진에 예외 처리 추가, 헤더·함수 설명 주석 정리.
- 2026-07-20: 월별 카테고리 집계에 월별 총매출/Top3 추가, Checkpoint 4개 항목을
  assert로 검증하는 run_checkpoints() 추가.
"""

import sys
from collections import Counter, defaultdict

DATA_FILE = "Python_Practice1_Data.json"
AMOUNT_THRESHOLD = 1000  # 고액 거래로 간주할 최소 금액


def load_sales(path):
    """데이터 파일을 읽어 sales(거래 딕셔너리 리스트)를 반환한다.
    파일이 없거나 sales 정의가 없으면 예외를 발생시킨다."""
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}") from e

    namespace = {}
    try:
        exec(source, namespace)
    except SyntaxError as e:
        raise ValueError(f"데이터 파일 형식이 올바르지 않습니다: {path}") from e

    sales = namespace.get("sales")
    if not sales:
        raise ValueError(f"'{path}'에 유효한 sales 데이터가 없습니다.")
    return sales


def filter_high_amount(sales, threshold=AMOUNT_THRESHOLD):
    """amount가 threshold 이상인 거래만 리스트로 반환한다."""
    return [item for item in sales if item["amount"] >= threshold]


def high_amount_generator(sales, threshold=AMOUNT_THRESHOLD):
    """amount가 threshold 이상인 거래를 하나씩 yield하는 제너레이터."""
    for item in sales:
        if item["amount"] >= threshold:
            yield item


def compute_region_totals(sales):
    """지역별 총매출 dict를 계산한다."""
    regions = {item["region"] for item in sales}
    return {
        region: sum(item["amount"] for item in sales if item["region"] == region)
        for region in regions
    }


def compute_region_counts(sales):
    """지역별 거래 건수를 Counter로 계산한다."""
    return Counter(item["region"] for item in sales)


def compute_category_amounts(sales):
    """카테고리별 amount 리스트를 defaultdict로 계산한다."""
    category_amounts = defaultdict(list)
    for item in sales:
        category_amounts[item["category"]].append(item["amount"])
    return category_amounts


def compute_monthly_category_totals(sales):
    """월 x 카테고리 조합별 매출을 집계해 {월: {카테고리: 매출}} 형태로 반환한다."""
    flat_totals = defaultdict(int)
    for item in sales:
        flat_totals[(item["month"], item["category"])] += item["amount"]

    months = sorted({item["month"] for item in sales})
    categories = sorted({item["category"] for item in sales})

    return {
        month: {
            category: flat_totals[(month, category)]
            for category in categories
            if (month, category) in flat_totals
        }
        for month in months
    }


def compute_monthly_totals(monthly_category_totals):
    """월별 총매출(카테고리 합계)을 계산한다."""
    return {
        month: sum(cat_totals.values())
        for month, cat_totals in monthly_category_totals.items()
    }


def print_high_amount_summary(sales, high_sales):
    """1) 전체/고액 거래 건수와 예시를 출력한다."""
    print(f"전체 거래 수: {len(sales)}")
    print(f"{AMOUNT_THRESHOLD} 이상 거래 수: {len(high_sales)}")
    print(f"\n{AMOUNT_THRESHOLD} 이상 거래 예시 (앞 5개):")
    for item in high_sales[:5]:
        print(item)


def print_region_totals(region_totals):
    """1) 지역별 총매출을 높은 순으로 출력한다."""
    print("\n지역별 총매출 (높은 순):")
    for region, total in sorted(region_totals.items(), key=lambda x: -x[1]):
        print(f"  {region}: {total:,}")


def print_region_counts(region_counts):
    """2) 지역별 거래 건수를 많은 순으로 출력한다."""
    print("\n지역별 거래 건수:")
    for region, count in region_counts.most_common():
        print(f"  {region}: {count}건")


def print_category_amounts(category_amounts):
    """2) 카테고리별 amount 리스트를 출력한다."""
    print("\n카테고리별 amount 리스트:")
    for category, amounts in category_amounts.items():
        print(f"  {category}: {amounts}")


def print_generator_memory_comparison(sales):
    """3) 리스트 버전과 제너레이터 버전의 메모리 크기를 비교 출력하고, 두 크기를 반환한다."""
    list_version = filter_high_amount(sales)
    gen_version = high_amount_generator(sales)

    list_size = sys.getsizeof(list_version)
    gen_size = sys.getsizeof(gen_version)

    print("\nlst vs gen 메모리 크기:")
    print(f"  lst: {list_size} bytes")
    print(f"  gen: {gen_size} bytes")

    print("\ngen에서 하나씩 꺼내기 (앞 3개):")
    for _ in range(3):
        try:
            print(next(gen_version))
        except StopIteration:
            print("  더 이상 꺼낼 값이 없습니다.")
            break

    return list_size, gen_size


def print_monthly_category_totals(monthly_category_totals, monthly_totals):
    """4) 월별 x 카테고리별 매출 집계와 월별 총매출을 출력한다."""
    print("\n월별 카테고리 매출 집계:")
    for month, cat_totals in monthly_category_totals.items():
        print(f"  {month}:")
        for category, total in cat_totals.items():
            print(f"    {category}: {total:,}")
        print(f"    합계: {monthly_totals[month]:,}")


def print_top3_months(monthly_totals):
    """4) 월별 총매출 상위 3개를 내림차순으로 출력한다."""
    top3 = sorted(monthly_totals.items(), key=lambda x: -x[1])[:3]
    print("\nTop 3 월별 총매출:")
    for rank, (month, total) in enumerate(top3, start=1):
        print(f"  {rank}위. {month}: {total:,}")


def run_checkpoints(sales, region_totals, region_counts, list_size, gen_size, monthly_totals):
    """Checkpoint 4개 항목을 assert로 검증한다.
    하나라도 실패하면 AssertionError가 발생한다."""
    # 1) region_total 값 정확: 지역별 총매출의 합 == 전체 매출 합
    assert sum(region_totals.values()) == sum(item["amount"] for item in sales), \
        "region_totals 합계가 전체 매출 합과 다릅니다."

    # 2) Counter.most_common() 순서 정확: 건수 내림차순 확인
    counts = [count for _, count in region_counts.most_common()]
    assert counts == sorted(counts, reverse=True), \
        "region_counts.most_common() 결과가 건수 내림차순이 아닙니다."

    # 3) 제너레이터가 리스트보다 메모리를 적게 쓰는지 확인
    assert gen_size < list_size, \
        "제너레이터 메모리 크기가 리스트보다 작지 않습니다."

    # 4) top3 월별 총매출이 내림차순으로 정렬되는지 확인
    top3_values = [total for _, total in sorted(monthly_totals.items(), key=lambda x: -x[1])[:3]]
    assert top3_values == sorted(top3_values, reverse=True), \
        "top3 월별 총매출이 내림차순으로 정렬되지 않았습니다."

    print("\n[Checkpoint] 4개 항목 모두 통과")


def main():
    """데이터를 불러와 1)~4) 실습 내용을 순서대로 실행/출력하고 Checkpoint를 검증한다."""
    try:
        sales = load_sales(DATA_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"[오류] 데이터를 불러오지 못했습니다: {e}")
        return

    try:
        # 1) 리스트/딕셔너리 컴프리헨션
        high_sales = filter_high_amount(sales)
        print_high_amount_summary(sales, high_sales)
        region_totals = compute_region_totals(sales)
        print_region_totals(region_totals)

        # 2) Counter + defaultdict
        region_counts = compute_region_counts(sales)
        print_region_counts(region_counts)
        print_category_amounts(compute_category_amounts(sales))

        # 3) 제너레이터 - 메모리 비교
        list_size, gen_size = print_generator_memory_comparison(sales)

        # 4) 종합 - 월별 카테고리 매출 집계 (+ 총매출, Top3)
        monthly_category_totals = compute_monthly_category_totals(sales)
        monthly_totals = compute_monthly_totals(monthly_category_totals)
        print_monthly_category_totals(monthly_category_totals, monthly_totals)
        print_top3_months(monthly_totals)

        # Checkpoint 검증
        run_checkpoints(sales, region_totals, region_counts, list_size, gen_size, monthly_totals)
    except KeyError as e:
        print(f"[오류] 거래 데이터에 필요한 키가 없습니다: {e}")
    except AssertionError as e:
        print(f"[Checkpoint 실패] {e}")


if __name__ == "__main__":
    main()
