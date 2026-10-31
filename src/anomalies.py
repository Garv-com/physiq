from src.kalman import weight_innovations

WARMUP_POINTS = 7


def detect_weight_anomalies(
    df,
    threshold=3.0,
    warmup_points=WARMUP_POINTS,
    date_col="date",
    weight_col="weight",
):
    innovations = weight_innovations(df, date_col=date_col, weight_col=weight_col)
    innovations = innovations.iloc[warmup_points:]
    flagged = innovations[innovations["z_score"].abs() >= threshold]

    return [
        {
            "date": row["date"],
            "weight": round(row["weight"], 2),
            "expected": round(row["expected"], 2),
            "difference_kg": round(row["weight"] - row["expected"], 2),
            "z_score": round(row["z_score"], 2),
            "direction": "high" if row["z_score"] > 0 else "low",
        }
        for _, row in flagged.iterrows()
    ]