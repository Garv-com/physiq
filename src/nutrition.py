import numpy as np
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