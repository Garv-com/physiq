import numpy as np
import pandas as pd


def compute_ema_trend(df, value_col="weight", alpha=0.1, output_col="trend_weight"):
    values = df[value_col].to_numpy()
    trend = np.zeros(len(values))
    trend[0] = values[0]

    for i in range(1, len(values)):
        trend[i] = alpha * values[i] + (1 - alpha) * trend[i - 1]

    result = df.copy()
    result[output_col] = trend.round(3)

    return result


def compute_time_aware_ema(df, date_col="date", value_col="weight", alpha=0.1, output_col="trend_weight"):
    result = df.copy()
    result["_parsed_date"] = pd.to_datetime(result[date_col])
    result = result.sort_values("_parsed_date").reset_index(drop=True)

    values = result[value_col].to_numpy(dtype=float)
    day_numbers = (result["_parsed_date"] - result["_parsed_date"].iloc[0]).dt.days.to_numpy()

    trend = np.zeros(len(values))
    trend[0] = values[0]

    for i in range(1, len(values)):
        gap = day_numbers[i] - day_numbers[i - 1]
        effective_alpha = alpha if gap == 1 else 1 - (1 - alpha) ** gap
        trend[i] = effective_alpha * values[i] + (1 - effective_alpha) * trend[i - 1]

    result[output_col] = trend.round(3)

    return result.drop(columns="_parsed_date")