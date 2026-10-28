import numpy as np
import pandas as pd
from scipy import stats


def _recent_window(df, date_col, weight_col, lookback_days, min_points):
    dates = pd.to_datetime(df[date_col])
    last_date = dates.max()
    in_window = dates >= last_date - pd.Timedelta(days=lookback_days - 1)

    x = (dates[in_window] - last_date).dt.days.to_numpy(dtype=float)
    y = df.loc[in_window, weight_col].to_numpy(dtype=float)

    if len(x) < min_points:
        raise ValueError(
            f"Need at least {min_points} weigh-ins in the last {lookback_days} days; found {len(x)}."
        )

    return x, y


def _fit_line(x, y):
    n = len(x)
    x_mean = x.mean()
    sxx = ((x - x_mean) ** 2).sum()
    slope = ((x - x_mean) * (y - y.mean())).sum() / sxx
    intercept = y.mean() - slope * x_mean
    residuals = y - (intercept + slope * x)
    dof = n - 2
    resid_std = np.sqrt((residuals ** 2).sum() / dof)
    return slope, intercept, resid_std, sxx, x_mean, n, dof


def forecast_weight(
    df,
    days_ahead=14,
    lookback_days=28,
    confidence=0.9,
    min_points=10,
    date_col="date",
    weight_col="weight",
):
    x, y = _recent_window(df, date_col, weight_col, lookback_days, min_points)
    slope, intercept, resid_std, sxx, x_mean, n, dof = _fit_line(x, y)

    t_value = stats.t.ppf(0.5 + confidence / 2, dof)
    se_prediction = resid_std * np.sqrt(1 + 1 / n + (days_ahead - x_mean) ** 2 / sxx)
    predicted = intercept + slope * days_ahead

    return {
        "current_weight": round(float(intercept), 2),
        "forecast_weight": round(float(predicted), 2),
        "lower": round(float(predicted - t_value * se_prediction), 2),
        "upper": round(float(predicted + t_value * se_prediction), 2),
        "weekly_rate": round(float(slope * 7), 3),
        "days_ahead": days_ahead,
        "confidence": confidence,
        "points_used": int(n),
    }


def trend_status(
    df,
    window_days=28,
    margin=0.2,
    confidence=0.95,
    min_points=10,
    date_col="date",
    weight_col="weight",
):
    x, y = _recent_window(df, date_col, weight_col, window_days, min_points)
    slope, intercept, resid_std, sxx, x_mean, n, dof = _fit_line(x, y)

    t_value = stats.t.ppf(0.5 + confidence / 2, dof)
    slope_se = resid_std / np.sqrt(sxx)
    weekly = slope * 7
    low = weekly - t_value * slope_se * 7
    high = weekly + t_value * slope_se * 7

    if low >= -margin and high <= margin:
        status = "stable"
    elif low > 0:
        status = "gaining"
    elif high < 0:
        status = "losing"
    else:
        status = "uncertain"

    return {
        "status": status,
        "weekly_rate": round(float(weekly), 3),
        "weekly_rate_low": round(float(low), 3),
        "weekly_rate_high": round(float(high), 3),
        "margin": margin,
        "confidence": confidence,
        "points_used": int(n),
    }


def backtest_forecast(
    df,
    horizon=14,
    lookback_days=28,
    confidence=0.9,
    min_points=10,
    date_col="date",
    weight_col="weight",
):
    data = df.copy()
    data["_date"] = pd.to_datetime(data[date_col])
    data = data.sort_values("_date").reset_index(drop=True)

    rows = []
    for i in range(len(data)):
        origin = data["_date"].iloc[i]
        target = data[data["_date"] == origin + pd.Timedelta(days=horizon)]
        if target.empty:
            continue

        history = data[data["_date"] <= origin]
        try:
            result = forecast_weight(
                history,
                days_ahead=horizon,
                lookback_days=lookback_days,
                confidence=confidence,
                min_points=min_points,
                date_col=date_col,
                weight_col=weight_col,
            )
        except ValueError:
            continue

        recent = history[history["_date"] >= origin - pd.Timedelta(days=6)]
        actual = float(target[weight_col].iloc[0])

        rows.append(
            {
                "origin": origin,
                "actual": actual,
                "forecast": result["forecast_weight"],
                "lower": result["lower"],
                "upper": result["upper"],
                "baseline": float(recent[weight_col].mean()),
            }
        )

    out = pd.DataFrame(rows)
    out["inside_interval"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    out["model_abs_error"] = (out["actual"] - out["forecast"]).abs()
    out["baseline_abs_error"] = (out["actual"] - out["baseline"]).abs()
    return out