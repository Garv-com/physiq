#using exponential moving average to calculate the exact weight vs noise

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