import sys
sys.path.append(".")

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.smoothing import compute_ema_trend
from src.nutrition import estimate_maintenance
from src.training import build_exercise_summary


app = FastAPI(title="Physiq API")


@app.get("/")
def root():
    return {"status": "ok", "message": "Physiq API is running"}


@app.get("/nutrition/maintenance")
def get_maintenance_estimate():
    df = pd.read_csv("data/raw/synthetic_nutrition_log.csv")
    df_smoothed = compute_ema_trend(df, alpha=0.1)
    maintenance = estimate_maintenance(df_smoothed)

    return {
        "estimated_maintenance_kcal": maintenance,
        "days_used": len(df),
    }


@app.get("/training/summary/{exercise}")
def get_training_summary(exercise: str):
    df = pd.read_csv("data/raw/synthetic_workout_log.csv")

    valid_exercises = df["exercise"].unique().tolist()
    if exercise not in valid_exercises:
        raise HTTPException(status_code=404, detail=f"Exercise not found. Valid options: {valid_exercises}")

    summary = build_exercise_summary(df, exercise)
    summary["total_volume"] = summary["total_volume"].round(1)
    summary["date"] = summary["date"].astype(str)

    return summary.to_dict(orient="records")