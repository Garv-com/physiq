import numpy as np
import pandas as pd


def generate_weight_data(
    days=90,
    start_weight=75.0,
    true_maintenance=2400,
    calorie_mean=2400,
    calorie_std=250,
    kcal_per_kg=7700,
    weight_noise_std=0.3,
    seed=42,
):
    rng = np.random.default_rng(seed)       #creates an object

    dates = pd.date_range(start="2026-01-01", periods=days, freq="D")       #array of 90 days
    calories = rng.normal(loc=calorie_mean, scale=calorie_std, size=days)       #random cal for all 90 days with mean=calorie_mean and std deviation = cal_std

    true_weight = np.zeros(days)    #array of 90 zeros
    true_weight[0] = start_weight       

    for i in range(1, days):
        calorie_balance = calories[i - 1] - true_maintenance
        daily_change = calorie_balance / kcal_per_kg
        true_weight[i] = true_weight[i - 1] + daily_change

    observed_noise = rng.normal(loc=0, scale=weight_noise_std, size=days)
    observed_weight = true_weight + observed_noise

    df = pd.DataFrame({
        "date": dates,
        "calories": calories.round(0),
        "true_weight": true_weight.round(3),
        "weight": observed_weight.round(2),
    })

    return df


if __name__ == "__main__":
    df = generate_weight_data()
    df.to_csv("data/raw/synthetic_nutrition_log.csv", index=False)
    print(df.head(10))
    print(f"\nGenerated {len(df)} days of data.")