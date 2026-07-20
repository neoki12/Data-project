"""
작성자   : 광주캠퍼스_2반_김현수
최초 작성일 : 2026-07-20
수정일   : 2026-07-20
목적     : 매출 데이터를 안전하게 읽고, pydantic으로 검증한 뒤, 결과를 파일로
          저장 및 재로딩까지 확인하는 실습 프로그램

구성
---------- 
1) 예외 처리 + 파일 읽기       : safe_load_csv() - try/except/else/finally
2) Pydantic v2 스키마 정의     : SalesRecord
3) 검증 파이프라인             : validate_sales() - valid/errors 분리
4) 결과 파일 저장 + 재로딩 확인 : CSV/JSON 저장 후 재로딩하여 건수 검증

변경 내역
----------
- 2026-07-20: logging(콘솔+파일) 설정, safe_load_csv/SalesRecord/validate_sales/
  저장·재로딩 함수 작성.
- 2026-07-20: import를 최상단으로 통합, Checkpoint 4개 항목을 고정된 테스트
  데이터(valid 4건/errors 3건)로 assert 검증하는 run_checkpoints() 추가.

파일 저장 방식 (실행할 때마다)
----------
- valid_sales.csv, errors.json, checkpoint_valid.csv, checkpoint_errors.json
  : open(path, "w", ...)로 열기 때문에 매번 "덮어쓰기" - 이전 실행 결과는 사라지고
    최신 실행 결과로 교체된다. 새 파일명으로 계속 쌓이지 않는다.
- app.log
  : logging.FileHandler는 mode를 지정하지 않으면 기본값이 "a"(이어쓰기) -
    실행할 때마다 기존 로그를 지우지 않고 파일 끝에 새 로그를 계속 추가한다.
"""

import csv
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

# ---------------- 로깅 설정 (콘솔 + 파일 동시 기록) ----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

ch = logging.StreamHandler()       # 콘솔 출력용 핸들러
fh = logging.FileHandler("app.log", encoding="utf-8")  # 파일 출력용 핸들러
ch.setFormatter(fmt)
fh.setFormatter(fmt)

logger.addHandler(ch)
logger.addHandler(fh)

DATA_FILE = "Python_Practice2_Data.json"
VALID_OUTPUT_FILE = "valid_sales.csv"
ERRORS_OUTPUT_FILE = "errors.json"


# ---------------- 1) 예외 처리 + 파일 읽기 ----------------

def safe_load_csv(path=DATA_FILE):
    """path의 데이터 파일을 읽어 dict 리스트를 반환한다.
    파일이 없거나 형식이 잘못되면 None을 반환한다."""
    sales = None
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()

        namespace = {}
        exec(source, namespace)
        sales = namespace.get("sales")
        if not sales:
            raise ValueError(f"'{path}'에 유효한 데이터가 없습니다.")
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {path}")
        return None
    except (SyntaxError, ValueError) as e:
        logger.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None
    else:
        logger.info(f"'{path}'에서 {len(sales)}건의 데이터를 불러왔습니다.")
        return sales
    finally:
        print("로딩 종료")


# ---------------- 2) Pydantic v2 스키마 정의 ----------------

class SalesRecord(BaseModel):
    """매출 거래 한 건에 대한 스키마.
    month·region은 비어있으면 안 되고, amount는 0보다 커야 한다.
    category는 없어도 된다."""

    month: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    category: Optional[str] = None


# ---------------- 3) 검증 파이프라인 (valid / errors 분리) ----------------

def validate_sales(raw_data):
    """raw_data(dict 리스트)의 각 row를 SalesRecord로 검증한다.
    성공한 row는 SalesRecord 객체로 valid 리스트에,
    실패한 row는 {"row": 원본 데이터, "error": 오류 메시지} 형태로
    errors 리스트에 담아 (valid, errors) 튜플로 반환한다."""
    valid = []
    errors = []
    for row in raw_data:
        try:
            record = SalesRecord(**row)
        except ValidationError as e:
            errors.append({"row": row, "error": str(e)})
        else:
            valid.append(record)
    return valid, errors


# ---------------- 4) 결과 파일 저장 + 재로딩 확인 ----------------

def save_valid_to_csv(valid_records, path=VALID_OUTPUT_FILE):
    """valid SalesRecord 리스트를 CSV 파일로 저장한다."""
    fieldnames = list(SalesRecord.model_fields.keys())
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in valid_records:
                writer.writerow(record.model_dump())
    except OSError as e:
        logger.error(f"CSV 저장 실패: {e}")
        raise
    logger.info(f"valid {len(valid_records)}건을 '{path}'에 저장했습니다.")


def save_errors_to_json(error_rows, path=ERRORS_OUTPUT_FILE):
    """errors 리스트를 JSON 파일로 저장한다."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(error_rows, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"JSON 저장 실패: {e}")
        raise
    logger.info(f"errors {len(error_rows)}건을 '{path}'에 저장했습니다.")


def reload_and_verify(valid_records, error_rows, csv_path=VALID_OUTPUT_FILE, json_path=ERRORS_OUTPUT_FILE):
    """저장한 CSV/JSON 파일을 다시 읽어 건수가 원본과 일치하는지 검증한다."""
    try:
        with open(csv_path, encoding="utf-8") as f:
            reloaded_valid = list(csv.DictReader(f))
        with open(json_path, encoding="utf-8") as f:
            reloaded_errors = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"재로딩할 파일을 찾을 수 없습니다: {e}")
        raise

    assert len(reloaded_valid) == len(valid_records), \
        f"CSV 재로딩 건수 불일치: {len(reloaded_valid)} != {len(valid_records)}"
    assert len(reloaded_errors) == len(error_rows), \
        f"JSON 재로딩 건수 불일치: {len(reloaded_errors)} != {len(error_rows)}"

    return reloaded_valid, reloaded_errors


# ---------------- Checkpoint 검증 ----------------
# valid 4건 + errors 3건이 되도록 고정한 테스트 데이터.
# 실제 운영 데이터(main)와 분리해서, 매번 같은 값으로 assert할 수 있게 한다.
CHECKPOINT_DATA = [
    {"month": "2024-01", "region": "서울", "amount": 1500, "category": "전자"},
    {"month": "2024-02", "region": "부산", "amount": 800},
    {"month": "2024-03", "region": "대구", "amount": 1200, "category": "의류"},
    {"month": "2024-04", "region": "인천", "amount": 950, "category": "식품"},
    {"month": "", "region": "서울", "amount": 1000, "category": "전자"},   # month 빈 값
    {"month": "2024-05", "region": "", "amount": 500},                     # region 빈 값
    {"month": "2024-06", "region": "광주", "amount": -100},                # amount 0 이하
]


def run_checkpoints():
    """과제 Checkpoint 4개 항목을 assert로 검증한다.
    하나라도 실패하면 AssertionError가 발생한다."""
    print("\n---------- Checkpoint ----------")

    # 1) safe_load_csv 동작: 없는 파일이면 None을 반환하는지 확인
    assert safe_load_csv("존재하지_않는_파일.json") is None, \
        "safe_load_csv가 없는 파일에 대해 None을 반환하지 않습니다."
    print("[1] safe_load_csv + 없는 파일 -> None 확인 완료")

    # 2) ValidationError 발생 시 오류 내용이 출력되는지 확인
    try:
        SalesRecord(month="", region="서울", amount=-100)
    except ValidationError as e:
        print("[2] ValidationError 오류 내용:")
        print(e)

    # 3) 고정 테스트 데이터로 valid 4건 / errors 3건 확인
    valid_records, error_rows = validate_sales(CHECKPOINT_DATA)
    assert len(valid_records) == 4, f"valid 건수가 4가 아닙니다: {len(valid_records)}"
    assert len(error_rows) == 3, f"errors 건수가 3이 아닙니다: {len(error_rows)}"
    print(f"[3] valid {len(valid_records)}건 / errors {len(error_rows)}건 확인 완료")

    # 4) 저장 후 재로딩한 valid 건수가 4건인지 확인
    save_valid_to_csv(valid_records, path="checkpoint_valid.csv")
    save_errors_to_json(error_rows, path="checkpoint_errors.json")
    reloaded_valid, _ = reload_and_verify(
        valid_records, error_rows,
        csv_path="checkpoint_valid.csv", json_path="checkpoint_errors.json",
    )
    assert len(reloaded_valid) == 4, f"재로딩된 valid 건수가 4가 아닙니다: {len(reloaded_valid)}"
    print(f"[4] 재로딩 후 len(reloaded)==4 확인 완료")

    print("\n[Checkpoint] 4개 항목 모두 통과")


def main():
    """실제 데이터를 로딩 -> 검증 -> 저장 -> 재로딩까지 실행하고 Checkpoint를 검증한다."""
    raw_data = safe_load_csv()
    if raw_data is None:
        return

    valid_records, error_rows = validate_sales(raw_data)
    print(f"\n전체 {len(raw_data)}건 중 valid {len(valid_records)}건 / errors {len(error_rows)}건")

    save_valid_to_csv(valid_records)
    save_errors_to_json(error_rows)
    reload_and_verify(valid_records, error_rows)

    try:
        run_checkpoints()
    except AssertionError as e:
        print(f"\n[Checkpoint 실패] {e}")


if __name__ == "__main__":
    main()
