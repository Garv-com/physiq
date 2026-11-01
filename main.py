import sys
sys.path.append(".")

import datetime
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.anomalies import detect_weight_anomalies
from src.assistant import AssistantError, chat as assistant_chat, weekly_report as assistant_weekly_report
from src.database import (
    init_db,
    load_nutrition_df,
    load_recent_nutrition,
    load_workout_df,
    add_nutrition_entry,
    add_workout_set,
    seed_from_csv,
)
from src.forecasting import forecast_path, forecast_weight, trend_status
from src.limits import ai_budget, ai_limit_per_hour, demo_mode
from src.smoothing import compute_time_aware_ema
from src.nutrition import (
    MIN_COVERAGE,
    WARMUP_DAYS,
    estimate_maintenance,
    estimate_maintenance_rolling,
    logging_coverage,
)
from src.training import build_exercise_summary

ROOT = Path(__file__).resolve().parent
MIN_NUTRITION_DAYS = WARMUP_DAYS + 14

init_db()


def seed_demo_data():
    if not demo_mode():
        return
    nutrition_csv = ROOT / "data" / "raw" / "synthetic_nutrition_log.csv"
    workout_csv = ROOT / "data" / "raw" / "synthetic_workout_log.csv"
    if nutrition_csv.exists() and workout_csv.exists():
        seed_from_csv(str(nutrition_csv), str(workout_csv))


@asynccontextmanager
async def lifespan(app):
    seed_demo_data()
    yield


def block_in_demo():
    if demo_mode():
        raise HTTPException(status_code=403, detail="Logging is disabled in the public demo.")


def check_ai_budget():
    if demo_mode() and not ai_budget.try_spend(ai_limit_per_hour()):
        raise HTTPException(
            status_code=429,
            detail="The public demo has reached its hourly AI limit. Try again later.",
        )


app = FastAPI(title="Physiq API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",")
        if origin.strip()
    ],
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


@app.get("/health")
def health():
    return {"status": "ok", "message": "Physiq API is running", "demo_mode": demo_mode()}


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


@app.get("/nutrition/anomalies")
def get_weight_anomalies(threshold: float = Query(3.0, ge=2.0, le=6.0)):
    df = load_nutrition_df()
    try:
        return detect_weight_anomalies(df, threshold=threshold)
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
    block_in_demo()
    add_nutrition_entry(entry.date.isoformat(), entry.calories, entry.weight)
    return {"status": "saved", "date": entry.date.isoformat()}


@app.post("/training/sets", status_code=201)
def log_workout_set(entry: WorkoutSetEntry):
    block_in_demo()
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
    check_ai_budget()
    if request.messages[0].role != "user" or request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="The conversation must start and end with a user message.",
        )

    try:
        return assistant_chat([message.model_dump() for message in request.messages])
    except AssistantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail)


@app.post("/assistant/weekly-report")
def assistant_weekly_report_endpoint():
    check_ai_budget()
    try:
        return assistant_weekly_report()
    except AssistantError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail)


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")