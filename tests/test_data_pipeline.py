"""Offline synthetic tests: do not represent study outcomes."""
import numpy as np
import pandas as pd
import pytest

from weather_traffic.analysis import (
    TEMP, PRECIP, MONTHS, join_and_features, load_accidents, monthly_weather,
)


def test_monthly_weather_mean_across_stations_then_sum_daily():
    df = pd.DataFrame({
        "station": ["a", "b", "a", "b"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        TEMP: [0, 2, 4, 6], PRECIP: [3, 5, 7, 9],
    })
    result = monthly_weather(df)
    assert len(result) == 1
    assert result["mean_air_temperature"].iloc[0] == 3.0
    assert result["total_precipitation"].iloc[0] == 12.0


def test_trine_month_names_and_injury_only(tmp_path):
    csv = tmp_path / "acc.csv"
    csv.write_text("År;Måned;Dødsulykke;Ulykke med hardt skadde;Ulykke med lettere skadde\n"
                   "2024;Januar;1;2;3\n2024;Februar;0;1;4\n", encoding="utf-8")
    result = load_accidents(csv)
    assert result["injury_accidents"].tolist() == [6, 5]
    assert result["month"].tolist() == [1, 2]


def test_duplicate_months_are_rejected(tmp_path):
    csv = tmp_path / "dup.csv"
    csv.write_text("År;Måned;Dødsulykke;Ulykke med hardt skadde;Ulykke med lettere skadde\n"
                   "2024;Januar;1;2;3\n2024;Januar;0;0;1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple TRINE rows"):
        load_accidents(csv)


def test_join_sorted_and_cyclic_features():
    accident = pd.DataFrame({"year": [2024, 2024], "month": [2, 1],
                             "injury_accidents": [5, 6]})
    weather = pd.DataFrame({"year": [2024, 2024], "month": [1, 2],
                            "mean_air_temperature": [0, 1], "total_precipitation": [20, 30]})
    out = join_and_features(accident, weather)
    assert out["month"].tolist() == [1, 2]
    assert np.all(out["year_index"] == 0)
    assert np.isclose(out["month_sin"].iloc[0], 0.5)
