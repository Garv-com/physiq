import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    backtest_forecast,
    forecast_path,
    forecast_weight,
    forecast_weight_linear,
    trend_status,
)
from src.synthetic_generator import generate_weight_data


def linear_series(days, start=80.0, slope=-0.05):
    dates = pd.date_range("2026-01-01", periods=days)
    return pd.DataFrame({"date": dates, "weight": start + slope * np.arange(days)})


def test_linear_forecast_extrapolates_a_straight_line():
    df = linear_series(30)
    result = forecast_weight_linear(df, days_ahead=14)
    assert result["forecast_weight"] == pytest.approx(80.0 - 0.05 * (29 + 14), abs=0.01)
    assert result["weekly_rate"] == pytest.approx(-0.35, abs=0.001)
    assert result["method"] == "linear"


def test_kalman_forecast_follows_a_clear_trend():
    df = linear_series(60)
    result = forecast_weight(df, days_ahead=14)
    assert result["method"] == "kalman"
    assert result["forecast_weight"] == pytest.approx(80.0 - 0.05 * (59 + 14), abs=0.2)
    assert result["lower"] < result["forecast_weight"] < result["upper"]


def test_falls_back_to_linear_with_few_weigh_ins():
    df = linear_series(12)
    assert forecast_weight(df, days_ahead=7)["method"] == "linear"
    assert len(forecast_path(df, days_ahead=7)) == 8


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        forecast_weight(linear_series(5), days_ahead=7)
    with pytest.raises(ValueError):
        forecast_path(linear_series(5), days_ahead=7)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        forecast_weight(pd.DataFrame({"date": [], "weight": []}), days_ahead=7)


def test_path_starts_on_the_last_logged_day_and_uncertainty_grows():
    data = generate_weight_data(days=90, seed=2)
    path = forecast_path(data, days_ahead=14)

    assert len(path) == 15
    assert path[0]["date"] == data["date"].iloc[-1].strftime("%Y-%m-%d")
    assert path[-1]["date"] == (data["date"].iloc[-1] + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    first_width = path[0]["upper"] - path[0]["lower"]
    last_width = path[-1]["upper"] - path[-1]["lower"]
    assert last_width > first_width


def test_kalman_handles_missing_days():
    data = generate_weight_data(days=90, seed=2)
    rng = np.random.default_rng(1)
    keep = np.sort(rng.choice(len(data), size=65, replace=False))
    gappy = data.iloc[keep].reset_index(drop=True)
    result = forecast_weight(gappy, days_ahead=14)
    assert result["method"] == "kalman"
    assert result["lower"] < result["upper"]


def test_trend_status_labels():
    flat = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=28), "weight": 75.0})
    assert trend_status(flat)["status"] == "stable"
    assert trend_status(linear_series(28, slope=-0.1))["status"] == "losing"
    assert trend_status(linear_series(28, slope=0.1))["status"] == "gaining"


def test_trend_status_needs_enough_weigh_ins():
    with pytest.raises(ValueError):
        trend_status(linear_series(5))


def test_backtest_returns_forecasts_and_interval_flags():
    data = generate_weight_data(days=100, seed=4)
    result = backtest_forecast(data, horizon=14, step_days=5)
    assert len(result) > 5
    assert {"actual", "forecast", "lower", "upper", "inside_interval"} <= set(result.columns)
    assert 0.0 <= result["inside_interval"].mean() <= 1.0
    assert (result["lower"] <= result["upper"]).all()