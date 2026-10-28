import sys
sys.path.append(".")

import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.database import (
    init_db,
    load_nutrition_df,
    load_workout_df,
    add_nutrition_entry,
    add_workout_set,
)
from src.smoothing import compute_ema_trend
from src.nutrition import estimate_maintenance, estimate_maintenance_rolling
from src.training import build_exercise_summary

MIN_NUTRITION_DAYS = 14

init_db()

app = FastAPI(title="Physiq API")


class NutritionEntry(BaseModel):
    date: datetime.date
    calories: float = Field(gt=0, le=15000)
    weight: float = Field(gt=20, lt=500)


class WorkoutSetEntry(BaseModel):
    date: datetime.date
    exercise: str = Field(min_length=1, max_length=50)
    weight_kg: float = Field(ge=0, le=1000)
    reps: int = Field(ge=1, le=100)
    rpe: float | None = Field(default=None, ge=1, le=10)


@app.get("/")
def root():
    return {"status": "ok", "message": "Physiq API is running"}


@app.get("/nutrition/maintenance")
def get_maintenance_estimate():
    df = load_nutrition_df()
    if len(df) < MIN_NUTRITION_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_NUTRITION_DAYS} days of nutrition data are needed; found {len(df)}.",
        )

    df_smoothed = compute_ema_trend(df, alpha=0.1)
    maintenance = estimate_maintenance(df_smoothed)

    return {"estimated_maintenance_kcal": maintenance, "days_used": len(df)}


@app.get("/nutrition/maintenance/rolling")
def get_rolling_maintenance(window: int = Query(30, ge=7, le=365)):
    df = load_nutrition_df()
    if len(df) < window:
        raise HTTPException(
            status_code=400,
            detail=f"Window of {window} days requested, but only {len(df)} days of data exist.",
        )

    df_smoothed = compute_ema_trend(df, alpha=0.1)
    df_rolling = estimate_maintenance_rolling(df_smoothed, window=window)

    result = df_rolling[["date", "maintenance_estimate"]].dropna()
    return result.to_dict(orient="records")


@app.get("/training/summary/{exercise}")
def get_training_summary(exercise: str):
    df = load_workout_df()

    valid_exercises = df["exercise"].unique().tolist()
    if exercise not in valid_exercises:
        raise HTTPException(
            status_code=404,
            detail=f"Exercise not found. Valid options: {valid_exercises}",
        )

    summary = build_exercise_summary(df, exercise)
    summary["total_volume"] = summary["total_volume"].round(1)

    return summary.to_dict(orient="records")


@app.post("/nutrition/log")
def log_nutrition(entry: NutritionEntry):
    add_nutrition_entry(entry.date.isoformat(), entry.calories, entry.weight)
    return {"status": "saved", "date": entry.date.isoformat()}


@app.post("/training/sets", status_code=201)
def log_workout_set(entry: WorkoutSetEntry):
    exercise = entry.exercise.strip()
    if not exercise:
        raise HTTPException(status_code=422, detail="Exercise name cannot be blank.")

    result = add_workout_set(
        entry.date.isoformat(),
        exercise,
        entry.weight_kg,
        entry.reps,
        entry.rpe,
    )
    return {"status": "saved", **result}