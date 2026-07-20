"""
작성자   : 광주캠퍼스_2반_김현수
최초 작성일 : 2026-07-20
수정일   : 2026-07-20
목적     : [Day 1] 종합 실습 - 데이터 수집 미니 파이프라인
          3개 공공 API(Open-Meteo, Countries.dev, ip-api)를 asyncio + httpx로
          동시에 수집하고, Pydantic v2로 필드 타입/범위를 검증한 뒤
          CSV/Parquet 두 형식으로 저장해 읽기/쓰기 성능을 비교한다.

구성
----
1) 환경 준비        : requirements.txt (pip freeze로 버전 고정, venv는 별도 활성화)
2) 비동기 수집       : fetch_json() + collect_all() - asyncio.gather()로 3개 API 동시 호출
3) 스키마 검증       : WeatherHourRecord / CountryRecord / IPLocationRecord (Pydantic v2)
4) 저장 + 성능 비교  : save_and_compare() - CSV/Parquet 각각 저장·재로딩 시간 측정
5) 테스트           : tests/test_schema.py 참고 (pytest)

변경 내역
--------
2026-07-20
  1. requirements.txt 준비 - httpx/pandas/pyarrow/pytest/ruff 설치 후
     pip freeze로 버전 고정.
  2. curl로 3개 API(Open-Meteo/Countries.dev/ip-api) 실제 응답 구조를
     먼저 확인하고, 그 필드명을 그대로 파싱 코드에 반영.
  3. fetch_json() / collect_all() 작성 - asyncio.gather()로 3개 API를
     동시에 호출하고, API별 실패가 서로 영향 주지 않도록 개별 try/except 처리.
  4. WeatherHourRecord / CountryRecord / IPLocationRecord Pydantic v2
     모델과 parse_weather/parse_country/parse_ip 정의.
  5. save_and_compare()로 CSV/Parquet 저장·재로딩 시간을 측정하는
     성능 비교 로직 작성.
  6. Parquet 최초 호출 시 pyarrow 초기화 비용이 측정치를 왜곡하는 문제를
     발견해 _warm_up_parquet_engine()으로 수정.
  7. tests/test_schema.py 작성, pytest 6건 전부 통과 확인.
  8. ruff check / ruff format 적용해 코드 스타일 정리.
"""

import pandas as pd
import asyncio
import httpx
import logging
import time
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# ---------------- 로깅 설정 ----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_handler)

# ---------------- 상수 ----------------
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.5665&longitude=126.9780"
    "&hourly=temperature_2m,precipitation_probability"
    "&forecast_days=3&timezone=Asia/Seoul"
)
COUNTRY_URL = "https://countries.dev/alpha/KOR"
IP_URL = "http://ip-api.com/json/8.8.8.8"

DATA_DIR = Path(__file__).parent / "data"


# ---------------- 2) 비동기 수집 ----------------


async def fetch_json(
    client: httpx.AsyncClient, name: str, url: str
) -> tuple[str, Optional[dict]]:
    """API를 호출해 JSON을 반환한다. 실패하면 로그를 남기고 None을 반환한다."""
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return name, response.json()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.error(f"{name} 수집 실패: {e}")
        return name, None


async def collect_all() -> dict[str, Optional[dict]]:
    """3개 API를 asyncio.gather()로 동시에 수집해 {이름: JSON} dict로 반환한다."""
    sources = {"weather": WEATHER_URL, "country": COUNTRY_URL, "ip": IP_URL}
    async with httpx.AsyncClient() as client:
        tasks = [fetch_json(client, name, url) for name, url in sources.items()]
        results = await asyncio.gather(*tasks)
    return dict(results)


# ---------------- 3) 스키마 검증 (Pydantic v2) ----------------


class WeatherHourRecord(BaseModel):
    """Open-Meteo 시간대별 예보 한 건."""

    time: str = Field(min_length=1)
    temperature_c: float
    precipitation_probability: int = Field(ge=0, le=100)


class CountryRecord(BaseModel):
    """Countries.dev 국가 정보."""

    name: str = Field(min_length=1)
    capital: Optional[str] = None
    region: str = Field(min_length=1)
    population: int = Field(ge=0)
    area: float = Field(gt=0)


class IPLocationRecord(BaseModel):
    """ip-api IP 위치 정보."""

    query: str = Field(min_length=1)
    country: str = Field(min_length=1)
    region_name: Optional[str] = None
    city: Optional[str] = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


def parse_weather(raw: dict) -> list[WeatherHourRecord]:
    """Open-Meteo 응답의 hourly 배열을 시간별 레코드로 변환·검증한다."""
    hourly = raw.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    probs = hourly.get("precipitation_probability", [])

    records = []
    for t, temp, prob in zip(times, temps, probs):
        try:
            records.append(
                WeatherHourRecord(
                    time=t, temperature_c=temp, precipitation_probability=prob
                )
            )
        except ValidationError as e:
            logger.warning(f"weather 레코드 검증 실패 (time={t}): {e}")
    return records


def parse_country(raw: dict) -> Optional[CountryRecord]:
    """Countries.dev 응답에서 필요한 필드만 추출·검증한다."""
    try:
        return CountryRecord(
            name=raw.get("name", ""),
            capital=raw.get("capital"),
            region=raw.get("region", ""),
            population=raw.get("population", 0),
            area=raw.get("area", 0),
        )
    except ValidationError as e:
        logger.warning(f"country 레코드 검증 실패: {e}")
        return None


def parse_ip(raw: dict) -> Optional[IPLocationRecord]:
    """ip-api 응답에서 필요한 필드만 추출·검증한다."""
    try:
        return IPLocationRecord(
            query=raw.get("query", ""),
            country=raw.get("country", ""),
            region_name=raw.get("regionName"),
            city=raw.get("city"),
            lat=raw.get("lat", 0.0),
            lon=raw.get("lon", 0.0),
        )
    except ValidationError as e:
        logger.warning(f"ip 레코드 검증 실패: {e}")
        return None


# ---------------- 4) 저장 + 성능 비교 ----------------


def save_and_compare(df: pd.DataFrame, name: str, out_dir: Path) -> dict:
    """DataFrame을 CSV/Parquet로 각각 저장·재로딩하며 소요 시간을 측정한다."""
    csv_path = out_dir / f"{name}.csv"
    parquet_path = out_dir / f"{name}.parquet"

    t0 = time.perf_counter()
    df.to_csv(csv_path, index=False, encoding="utf-8")
    csv_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    df.to_parquet(parquet_path, index=False)
    parquet_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    reloaded_csv = pd.read_csv(csv_path)
    csv_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    reloaded_parquet = pd.read_parquet(parquet_path)
    parquet_read = time.perf_counter() - t0

    assert len(reloaded_csv) == len(df), f"{name}: CSV 재로딩 건수 불일치"
    assert len(reloaded_parquet) == len(df), f"{name}: Parquet 재로딩 건수 불일치"

    return {
        "name": name,
        "rows": len(df),
        "csv_write_ms": csv_write * 1000,
        "parquet_write_ms": parquet_write * 1000,
        "csv_read_ms": csv_read * 1000,
        "parquet_read_ms": parquet_read * 1000,
    }


def print_comparison(results: list[dict]) -> None:
    """CSV vs Parquet 성능 비교 결과를 출력한다."""
    print("\n=== CSV vs Parquet 성능 비교 ===")
    for r in results:
        print(f"[{r['name']}] rows={r['rows']}")
        print(
            f"  write : csv={r['csv_write_ms']:.2f}ms  parquet={r['parquet_write_ms']:.2f}ms"
        )
        print(
            f"  read  : csv={r['csv_read_ms']:.2f}ms  parquet={r['parquet_read_ms']:.2f}ms"
        )


# ---------------- 실행 ----------------


def _warm_up_parquet_engine(out_dir: Path) -> None:
    """pyarrow 엔진의 최초 호출 시 초기화 비용이 측정치를 왜곡하지 않도록
    본 측정 전에 더미 데이터로 한 번 미리 실행해둔다."""
    warm_path = out_dir / "_warmup.parquet"
    pd.DataFrame({"x": [0]}).to_parquet(warm_path, index=False)
    pd.read_parquet(warm_path)
    warm_path.unlink()


def main():
    """데이터 수집 -> 검증 -> 저장/성능비교 전체 파이프라인을 실행한다."""
    DATA_DIR.mkdir(exist_ok=True)

    logger.info("3개 API 비동기 수집 시작 (asyncio.gather)")
    try:
        raw = asyncio.run(collect_all())
    except Exception as e:
        logger.error(f"수집 단계에서 예기치 못한 오류 발생: {e}")
        return

    weather_records = parse_weather(raw["weather"]) if raw.get("weather") else []
    country_record = parse_country(raw["country"]) if raw.get("country") else None
    ip_record = parse_ip(raw["ip"]) if raw.get("ip") else None

    logger.info(f"weather: {len(weather_records)}건 검증 통과")
    logger.info(f"country: {'통과' if country_record else '실패'}")
    logger.info(f"ip: {'통과' if ip_record else '실패'}")

    results = []
    try:
        _warm_up_parquet_engine(DATA_DIR)
        if weather_records:
            df = pd.DataFrame([r.model_dump() for r in weather_records])
            results.append(save_and_compare(df, "weather", DATA_DIR))
        if country_record:
            df = pd.DataFrame([country_record.model_dump()])
            results.append(save_and_compare(df, "country", DATA_DIR))
        if ip_record:
            df = pd.DataFrame([ip_record.model_dump()])
            results.append(save_and_compare(df, "ip", DATA_DIR))
    except (OSError, AssertionError) as e:
        logger.error(f"저장/재로딩 단계에서 오류 발생: {e}")
        return

    print_comparison(results)


if __name__ == "__main__":
    main()
