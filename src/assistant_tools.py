import json

import pandas as pd

from src.anomalies import detect_weight_anomalies
from src.database import load_nutrition_df, load_recent_nutrition, load_workout_df
from src.forecasting import forecast_weight, trend_status
from src.nutrition import MIN_COVERAGE, WARMUP_DAYS, estimate_maintenance, logging_coverage
from src.smoothing import compute_time_aware_ema
from src.training import build_exercise_summary

MIN_NUTRITION_DAYS = WARMUP_DAYS + 14


def _clamp_int(value, low, high, default):
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


def _records(df):
    return json.loads(df.to_json(orient="records"))


def tool_get_recent_weight(days=14):
    days = _clamp_int(days, 1, 90, 14)
    df = load_recent_nutrition(days)
    if df.empty:
        return {"error": "No nutrition data has been logged yet."}
    return {"days_requested": days, "entries": _records(df)}


def tool_predict_weight(days=14):
    days = _clamp_int(days, 1, 60, 14)
    df = load_nutrition_df()
    try:
        return forecast_weight(df, days_ahead=days)
    except ValueError as error:
        return {"error": str(error)}


def tool_estimate_maintenance():
    df = load_nutrition_df()
    if len(df) < MIN_NUTRITION_DAYS:
        return {"error": f"At least {MIN_NUTRITION_DAYS} logged days are needed; found {len(df)}."}

    logged_days, span_days, coverage = logging_coverage(df)
    if coverage < MIN_COVERAGE:
        return {
            "error": (
                f"Only {logged_days} of {span_days} days are logged ({coverage:.0%}); "
                f"at least {MIN_COVERAGE:.0%} is needed."
            )
        }

    smoothed = compute_time_aware_ema(df, alpha=0.1)
    maintenance = estimate_maintenance(smoothed, warmup_days=WARMUP_DAYS)

    return {
        "estimated_maintenance_kcal": maintenance,
        "days_logged": logged_days,
        "days_span": span_days,
        "coverage": round(coverage, 2),
        "warmup_days_skipped": WARMUP_DAYS,
    }


def tool_detect_plateau():
    df = load_nutrition_df()
    try:
        return trend_status(df)
    except ValueError as error:
        return {"error": str(error)}


def tool_detect_anomalies(days=60):
    days = _clamp_int(days, 7, 90, 60)
    df = load_nutrition_df()
    try:
        anomalies = detect_weight_anomalies(df)
    except ValueError as error:
        return {"error": str(error)}

    last_date = pd.to_datetime(df["date"]).max()
    cutoff = (last_date - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    recent = [item for item in anomalies if item["date"] >= cutoff]

    return {
        "days_checked": days,
        "threshold_standard_deviations": 3.0,
        "count": len(recent),
        "anomalies": recent,
        "note": (
            "A flag means the weigh-in was far from what the trend predicted. It can be water, salt, "
            "a late meal or a scale error, and does not by itself mean a real change in body weight."
        ),
    }


def tool_analyze_progress(exercise):
    df = load_workout_df()
    valid_exercises = df["exercise"].unique().tolist()
    if exercise not in valid_exercises:
        return {"error": f"Unknown exercise. Valid options: {valid_exercises}"}

    summary = build_exercise_summary(df, exercise)
    summary["total_volume"] = summary["total_volume"].round(1)
    recent = summary.tail(6)

    return {
        "exercise": exercise,
        "sessions_logged": int(len(summary)),
        "recent_sessions": _records(recent),
        "trend_change_over_recent_sessions_kg": round(
            float(recent["trend_top_weight"].iloc[-1] - recent["trend_top_weight"].iloc[0]), 2
        ),
    }


def tool_get_model_explanation():
    return {
        "weight_trend": (
            "Time-aware exponential moving average of daily weigh-ins (alpha 0.1 per day, "
            "adjusted for missing days)."
        ),
        "maintenance_estimate": (
            "Slope of the smoothed weight trend after a 14-day warm-up, combined with mean logged "
            "calories using 7700 kcal per kg. Unlogged days are assumed to resemble logged days. "
            "Requires at least 70% of days logged."
        ),
        "weight_forecast": (
            "A local linear trend model (Kalman filter) fitted to the last 90 days of weigh-ins, "
            "which handles missing days and reports a 90% prediction interval. With fewer than 14 "
            "weigh-ins it falls back to a straight-line fit over the last 28 days. The 'method' "
            "field says which was used. It assumes the recent trend continues."
        ),
        "trend_status": (
            "Slope of the last 28 days with a 95% confidence interval. 'stable' means the whole "
            "interval lies within +/-0.2 kg per week; 'uncertain' means the data cannot tell."
        ),
        "anomaly_detection": (
            "A weigh-in is flagged when it is at least 3 standard deviations away from what the "
            "Kalman trend model predicted from the days before it."
        ),
        "training_progress": (
            "Heaviest set per session, smoothed with an exponential moving average (alpha 0.3). "
            "Estimated 1RM uses the Epley formula. A PR means a new best estimated 1RM."
        ),
        "validation_note": (
            "All validation was done on synthetic data with known ground truth, not on real users."
        ),
        "validated_performance_on_synthetic_data": {
            "maintenance_estimate": "Typically within about 15 to 25 kcal (roughly 1%) of the true value.",
            "forecast_interval": (
                "On held-out synthetic data the 90% interval contained the actual weigh-in about 90% "
                "to 94% of the time over 14-day horizons."
            ),
            "forecast_accuracy": (
                "The Kalman forecast had 3% to 11% lower error than a straight-line fit. When weight "
                "was flat it was about equal to a plain 7-day average; when weight was changing "
                "steadily it was far better than the 7-day average."
            ),
            "anomaly_detection": (
                "With day-to-day scale noise of 0.3 kg, it caught about 94% of 2 kg spikes, 62% of "
                "1.2 kg spikes and 22% of 0.8 kg spikes, with about 0.2 false alarms per 90 days. "
                "Smaller deviations are not reliably detectable."
            ),
            "session_top_weight": (
                "The heaviest set per session sits about 0.66 kg above true underlying strength on "
                "average, because it is the best of several noisy sets."
            ),
        },
        "known_limitations": [
            "Daily scale weight fluctuates from water and food, so short forecasts are noisy.",
            "Maintenance accuracy depends on honest, complete calorie logging.",
            "Forecasts do not know about future changes in diet or training.",
            "Sleep, stress and recovery are not tracked.",
            "Forecast validation used simulated data that suits the model, so real-world accuracy may be lower.",
        ],
    }


TOOL_FUNCTIONS = {
    "get_recent_weight": tool_get_recent_weight,
    "predict_weight": tool_predict_weight,
    "estimate_maintenance": tool_estimate_maintenance,
    "detect_plateau": tool_detect_plateau,
    "detect_anomalies": tool_detect_anomalies,
    "analyze_progress": tool_analyze_progress,
    "get_model_explanation": tool_get_model_explanation,
}

TOOL_SCHEMAS = [
    {
        "name": "get_recent_weight",
        "description": "Get the user's logged daily weights and calories for the last N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (1-90).", "minimum": 1, "maximum": 90}
            },
            "required": [],
        },
    },
    {
        "name": "predict_weight",
        "description": (
            "Forecast the user's scale weight N days after their last logged day, with a 90% "
            "prediction interval, current fitted weight and weekly rate of change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days ahead to forecast (1-60).", "minimum": 1, "maximum": 60}
            },
            "required": [],
        },
    },
    {
        "name": "estimate_maintenance",
        "description": "Estimate the user's maintenance calories (energy expenditure) from their logged weight and calories.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "detect_plateau",
        "description": (
            "Classify the user's recent weight trend as gaining, losing, stable or uncertain, "
            "with the weekly rate and its 95% confidence range."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "detect_anomalies",
        "description": (
            "List recent weigh-ins that were unusually far from what the trend model predicted, "
            "with the expected weight and how many standard deviations away each one was."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days to look back (7-90).", "minimum": 7, "maximum": 90}
            },
            "required": [],
        },
    },
    {
        "name": "analyze_progress",
        "description": (
            "Get recent training sessions for one exercise: top set, smoothed trend, volume and "
            "whether each session contained a personal record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exercise": {"type": "string", "description": "Exercise name, for example 'Squat' or 'Bench Press'."}
            },
            "required": ["exercise"],
        },
    },
    {
        "name": "get_model_explanation",
        "description": (
            "Get plain descriptions of how each model works, how it performed in validation, "
            "and its known limitations. Use when the user asks how reliable or how calculated something is."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


UNITS_NOTE = "Weights are in kilograms (kg), calories in kcal, rates in kg per week."


def run_tool(name, arguments):
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = function(**(arguments or {}))
    except TypeError:
        return {"error": "Invalid arguments for this tool."}
    except Exception as error:
        return {"error": f"Tool failed: {type(error).__name__}"}

    if isinstance(result, dict) and "error" not in result:
        result["units"] = UNITS_NOTE

    return result