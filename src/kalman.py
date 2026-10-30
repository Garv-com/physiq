import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

LOOKBACK_DAYS = 90
SLOPE_PRIOR_VAR = 0.0005
MIN_POINTS = 14
LOG_BOUNDS = [
    (math.log(1e-4), math.log(4.0)),
    (math.log(1e-8), math.log(1.0)),
    (math.log(1e-10), math.log(1e-2)),
]


def _daily_series(df, date_col, weight_col):
    data = df[[date_col, weight_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).drop_duplicates(date_col).set_index(date_col)
    full_index = pd.date_range(data.index.min(), data.index.max(), freq="D")
    return data[weight_col].reindex(full_index)


def _predict(level, slope, p11, p12, p22, q_level, q_slope):
    return (
        level + slope,
        slope,
        p11 + 2 * p12 + p22 + q_level,
        p12 + p22,
        p22 + q_slope,
    )


def _run_filter(observations, r, q_level, q_slope, slope_prior_var):
    level = observations[0]
    slope = 0.0
    p11, p12, p22 = 1.0, 0.0, slope_prior_var
    log_likelihood = 0.0

    for index, y in enumerate(observations):
        if index > 0:
            level, slope, p11, p12, p22 = _predict(level, slope, p11, p12, p22, q_level, q_slope)

        if math.isnan(y):
            continue

        innovation = y - level
        s = p11 + r
        k1 = p11 / s
        k2 = p12 / s
        log_likelihood += -0.5 * (math.log(2 * math.pi * s) + innovation * innovation / s)

        level = level + k1 * innovation
        slope = slope + k2 * innovation
        p22 = p22 - p12 * p12 / s
        p12 = (1 - k1) * p12
        p11 = (1 - k1) * p11

    return log_likelihood, level, slope, (p11, p12, p22)


def fit_local_linear_trend(observations, slope_prior_var=SLOPE_PRIOR_VAR):
    values = [float(v) for v in observations]
    known = np.array([v for v in values if not math.isnan(v)])
    diffs = np.diff(known)
    r0 = max(float(np.var(diffs)) / 2 if len(diffs) > 1 else 0.05, 1e-3)
    start = np.log([r0, 1e-4, 1e-7])

    def objective(log_params):
        r, q_level, q_slope = np.exp(log_params)
        return -_run_filter(values, r, q_level, q_slope, slope_prior_var)[0]

    result = minimize(objective, start, method="L-BFGS-B", bounds=LOG_BOUNDS)
    r, q_level, q_slope = (float(v) for v in np.exp(result.x))
    _, level, slope, covariance = _run_filter(values, r, q_level, q_slope, slope_prior_var)

    return {
        "r": r,
        "q_level": q_level,
        "q_slope": q_slope,
        "level": level,
        "slope": slope,
        "covariance": covariance,
    }


def _prepare(df, lookback_days, min_points, date_col, weight_col):
    series = _daily_series(df, date_col, weight_col).iloc[-lookback_days:]
    series = series.loc[series.first_valid_index():] if series.notna().any() else series
    count = int(series.notna().sum())

    if count < min_points:
        raise ValueError(
            f"Need at least {min_points} weigh-ins in the last {lookback_days} days; found {count}."
        )

    return series, count, fit_local_linear_trend(series.to_numpy())


def kalman_forecast_path(
    df,
    days_ahead=14,
    lookback_days=LOOKBACK_DAYS,
    confidence=0.9,
    min_points=MIN_POINTS,
    date_col="date",
    weight_col="weight",
):
    series, count, model = _prepare(df, lookback_days, min_points, date_col, weight_col)

    last_date = series.index[-1]
    z = float(norm.ppf(0.5 + confidence / 2))
    level, slope = model["level"], model["slope"]
    p11, p12, p22 = model["covariance"]

    path = []
    for step in range(0, days_ahead + 1):
        if step > 0:
            level, slope, p11, p12, p22 = _predict(
                level, slope, p11, p12, p22, model["q_level"], model["q_slope"]
            )
        spread = z * math.sqrt(p11 + model["r"])
        path.append(
            {
                "date": (last_date + pd.Timedelta(days=step)).strftime("%Y-%m-%d"),
                "forecast": round(level, 2),
                "lower": round(level - spread, 2),
                "upper": round(level + spread, 2),
            }
        )

    return path, model, count


def kalman_forecast(
    df,
    days_ahead=14,
    lookback_days=LOOKBACK_DAYS,
    confidence=0.9,
    min_points=MIN_POINTS,
    date_col="date",
    weight_col="weight",
):
    path, model, count = kalman_forecast_path(
        df, days_ahead, lookback_days, confidence, min_points, date_col, weight_col
    )

    return {
        "current_weight": path[0]["forecast"],
        "forecast_weight": path[-1]["forecast"],
        "lower": path[-1]["lower"],
        "upper": path[-1]["upper"],
        "weekly_rate": round(float(model["slope"]) * 7, 3),
        "days_ahead": days_ahead,
        "confidence": confidence,
        "points_used": count,
        "method": "kalman",
    }