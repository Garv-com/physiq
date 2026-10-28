import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MIN_COVERAGE = 0.7


def logging_coverage(df, date_col="date"):
    dates = pd.to_datetime(df[date_col])
    logged_days = int(dates.nunique())
    span_days = int((dates.max() - dates.min()).days) + 1
    return logged_days, span_days, logged_days / span_days


def estimate_maintenance(df, weight_col="trend_weight", calorie_col="calories", date_col="date", kcal_per_kg=7700):
    dates = pd.to_datetime(df[date_col])
    days = (dates - dates.min()).dt.days.to_numpy().reshape(-1, 1)
    weights = df[weight_col].to_numpy()

    model = LinearRegression()
    model.fit(days, weights)

    mean_daily_change = model.coef_[0]
    mean_calories = df[calorie_col].mean()

    maintenance = mean_calories - kcal_per_kg * mean_daily_change

    return round(maintenance, 1)


def estimate_maintenance_rolling(
    df,
    window=30,
    weight_col="trend_weight",
    calorie_col="calories",
    date_col="date",
    kcal_per_kg=7700,
    min_coverage=MIN_COVERAGE,
):
    result = df.copy()
    dates = pd.to_datetime(result[date_col])
    first_date = dates.min()
    min_points = max(2, int(np.ceil(min_coverage * window)))

    estimates = []
    for current in dates:
        window_start = current - pd.Timedelta(days=window - 1)

        if window_start < first_date:
            estimates.append(np.nan)
            continue

        window_df = result[(dates >= window_start) & (dates <= current)]

        if len(window_df) < min_points:
            estimates.append(np.nan)
            continue

        estimates.append(
            estimate_maintenance(
                window_df,
                weight_col=weight_col,
                calorie_col=calorie_col,
                date_col=date_col,
                kcal_per_kg=kcal_per_kg,
            )
        )

    result["maintenance_estimate"] = estimates

    return result