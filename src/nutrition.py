import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_maintenance(df, weight_col="trend_weight", calorie_col="calories", kcal_per_kg=7700):
    mean_calories = df[calorie_col].mean()

    days = np.arange(len(df)).reshape(-1, 1)
    weights = df[weight_col].to_numpy()

    model = LinearRegression()
    model.fit(days, weights)

    mean_daily_change = model.coef_[0]

    maintenance = mean_calories - kcal_per_kg * mean_daily_change

    return round(maintenance, 1)


def estimate_maintenance_rolling(df, window=30, weight_col="trend_weight", calorie_col="calories", kcal_per_kg=7700):
    estimates = [np.nan] * len(df)

    for i in range(window, len(df) + 1):
        window_df = df.iloc[i - window:i]
        estimates[i - 1] = estimate_maintenance(
            window_df, weight_col=weight_col, calorie_col=calorie_col, kcal_per_kg=kcal_per_kg
        )

    result = df.copy()
    result["maintenance_estimate"] = estimates

    return result