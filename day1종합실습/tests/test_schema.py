"""
day1종합실습.py의 Pydantic v2 스키마(WeatherHourRecord/CountryRecord/IPLocationRecord)가
요구된 타입·범위 규칙을 정확히 검증하는지 확인하는 pytest 테스트.
정상 케이스와 규칙 위반 케이스를 각각 확인한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from day1종합실습 import CountryRecord, IPLocationRecord, WeatherHourRecord


def test_weather_record_valid():
    record = WeatherHourRecord(
        time="2026-07-20T00:00", temperature_c=23.2, precipitation_probability=6
    )
    assert record.precipitation_probability == 6


def test_weather_record_rejects_out_of_range_probability():
    with pytest.raises(ValidationError):
        WeatherHourRecord(
            time="2026-07-20T00:00", temperature_c=23.2, precipitation_probability=150
        )


def test_country_record_valid():
    record = CountryRecord(
        name="Korea (Republic of)",
        capital="Seoul",
        region="Asia",
        population=51780579,
        area=100210,
    )
    assert record.population > 0


def test_country_record_rejects_negative_population():
    with pytest.raises(ValidationError):
        CountryRecord(name="Korea", region="Asia", population=-1, area=100210)


def test_ip_record_valid():
    record = IPLocationRecord(
        query="8.8.8.8", country="United States", lat=39.03, lon=-77.5
    )
    assert record.query == "8.8.8.8"


def test_ip_record_rejects_invalid_latitude():
    with pytest.raises(ValidationError):
        IPLocationRecord(query="8.8.8.8", country="US", lat=999, lon=-77.5)
