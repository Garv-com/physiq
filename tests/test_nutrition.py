import numpy as np
import pandas as pd
import pytest

from src.nutrition import estimate_maintenance, estimate_maintenance_rolling, logging_coverage
from src.smoothing import compute_ema_trend, compute_time_aware_ema
from src.synthetic_generator import generate_weight_data


def noiseless(calorie_mean, days=90, seed=1):
    return generate_weight_data(
        days=days,
        seed=seed,
        calorie_mean=calorie_mean,
        calorie_std=0,
        weight_noise_std=0,
        true_maintenance=2400,
    )


def row_index_estimate(df, kcal_per_kg=7700):
    slope = np.polyfit(np.arange(len(df)), df["trend_weight"].to_numpy(), 1)[0]
    return df["calories"].mean() - kcal_per_kg * slope


def test_flat_weight_means_maintenance_equals_intake():
    data = noiseless(2400)
    estimate = estimate_maintenance(compute_time_aware_ema(data, alpha=0.1), warmup_days=14)
    assert estimate == pytest.approx(2400, abs=2)


def test_recovers_true_maintenance_during_a_deficit():
    data = noiseless(2000)
    estimate = estimate_maintenance(compute_time_aware_ema(data, alpha=0.1), warmup_days=14)
    assert estimate == pytest.approx(2400, abs=15)


def test_missing_days_do_not_bias_the_estimate():
    data = noiseless(2000)
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(data), size=int(len(data) * 0.75), replace=False))
    gappy = data.iloc[keep].reset_index(drop=True)

    fixed = estimate_maintenance(compute_time_aware_ema(gappy, alpha=0.1), warmup_days=14)
    naive = row_index_estimate(compute_ema_trend(gappy, alpha=0.1))

    assert fixed == pytest.approx(2400, abs=40)
    assert abs(naive - 2400) > 60


def test_logging_coverage_counts_calendar_days():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-04"]})
    logged, span, coverage = logging_coverage(df)
    assert (logged, span) == (3, 4)
    assert coverage == 0.75


def test_warmup_longer_than_the_data_raises():
    data = noiseless(2400, days=10)
    smoothed = compute_time_aware_ema(data, alpha=0.1)
    with pytest.raises(ValueError):
        estimate_maintenance(smoothed, warmup_days=14)


def test_rolling_estimate_waits_for_a_full_window_and_matches_direct_estimate():
    data = noiseless(2000)
    smoothed = compute_time_aware_ema(data, alpha=0.1)
    rolling = estimate_maintenance_rolling(smoothed, window=30)

    assert rolling["maintenance_estimate"].iloc[:29].isna().all()
    assert not np.isnan(rolling["maintenance_estimate"].iloc[29])

    direct = estimate_maintenance(smoothed.iloc[-30:])
    assert rolling["maintenance_estimate"].iloc[-1] == direct


def test_rolling_estimate_skips_sparse_windows():
    data = noiseless(2000)
    smoothed = compute_time_aware_ema(data, alpha=0.1)
    sparse = smoothed.iloc[::3].reset_index(drop=True)
    rolling = estimate_maintenance_rolling(sparse, window=30)
    assert rolling["maintenance_estimate"].isna().all()