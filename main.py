import sys
sys.path.append(".")

import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.assistant import AssistantError, chat as assistant_chat
from src.database import (
    init_db,
    load_nutrition_df,
    load_recent_nutrition,
    load_workout_df,
    add_nutrition_entry,
    add_workout_set,
)
from src.forecasting import forecast_path, forecast_weight, trend_status
from src.smoothing import compute_time_aware_ema
from src.nutrition import (
    MIN_COVERAGE,
    WARMUP_DAYS,
    estimate_maintenance,
    estimate_maintenance_rolling,
    logging_coverage,
)
from src.training import build_exercise_summary

MIN_NUTRITION_DAYS = WARMUP_DAYS + 14

init_db()

app = FastAPI(title="Physiq API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)


@app.get("/")
def root():
    return {"status": "ok", "message": "Physiq API is running"}


@app.get("/nutrition/maintenance")
def get_maintenance_estimate():
    df = load_nutrition_df()
    if len(df) < MIN_NUTRITION_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_NUTRITION_DAYS} logged days are needed; found {len(df)}.",
        )

    logged_days, span_days, coverage = logging_coverage(df)
    if coverage < MIN_COVERAGE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {logged_days} of {span_days} days are logged ({coverage:.0%}). "
                f"At least {MIN_COVERAGE:.0%} is needed for a reliable estimate."
            ),
        )

    df_smoothed = compute_time_aware_ema(df, alpha=0.1)
    maintenance = estimate_maintenance(df_smoothed, warmup_days=WARMUP_DAYS)

    return {
        "estimated_maintenance_kcal": maintenance,
        "days_used": logged_days,
        "days_span": span_days,
        "coverage": round(coverage, 2),
        "warmup_days_skipped": WARMUP_DAYS,
    }


@app.get("/nutrition/maintenance/rolling")
def get_rolling_maintenance(window: int = Query(30, ge=7, le=365)):
    df = load_nutrition_df()
    if len(df) < MIN_NUTRITION_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_NUTRITION_DAYS} logged days are needed; found {len(df)}.",
        )

    df_smoothed = compute_time_aware_ema(df, alpha=0.1)
    df_rolling = estimate_maintenance_rolling(df_smoothed, window=window, warmup_days=WARMUP_DAYS)

    result = df_rolling[["date", "maintenance_estimate"]].dropna()
    if result.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No {window}-day window has enough logged days. Each window needs "
                f"{MIN_COVERAGE:.0%} coverage and starts after the first {WARMUP_DAYS} days."
            ),
        )

    return result.to_dict(orient="records")


@app.get("/nutrition/forecast")
def get_weight_forecast(days: int = Query(14, ge=1, le=60)):
    df = load_nutrition_df()
    try:
        return forecast_weight(df, days_ahead=days)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/nutrition/trend-status")
def get_trend_status():
    df = load_nutrition_df()
    try:
        return trend_status(df)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/nutrition/forecast/path")
def get_forecast_path(days: int = Query(14, ge=1, le=60)):
    df = load_nutrition_df()
    try:
        return forecast_path(df, days_ahead=days)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/nutrition/history")
def get_nutrition_history(days: int = Query(60, ge=7, le=365)):
    return load_recent_nutrition(days).to_dict(orient="records")


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


@app.post("/assistant/chat")
def assistant_chat_endpoint(request: ChatRequest):
    if request.messages[0].role != "user" or request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The conversation must start and end with a user message.",
        )

    try:
        return assistant_chat([message.model_dump() for message in request.messages])
    except AssistantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail)