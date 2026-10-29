import json

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
            "Straight-line fit to the last 28 days of raw weigh-ins, extended forward, with a 90% "
            "prediction interval. It assumes the recent trend continues."
        ),
        "trend_status": (
            "Slope of the last 28 days with a 95% confidence interval. 'stable' means the whole "
            "interval lies within +/-0.2 kg per week; 'uncertain' means the data cannot tell."
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
                "The 90% interval contained the actual weigh-in about 86% of the time over 14-day "
                "horizons, so it is slightly overconfident."
            ),
            "forecast_vs_baseline": (
                "When weight was flat, a plain 7-day average was a little more accurate than the "
                "trend forecast; when weight was changing steadily, the trend forecast was far better."
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
        ],
    }


TOOL_FUNCTIONS = {
    "get_recent_weight": tool_get_recent_weight,
    "predict_weight": tool_predict_weight,
    "estimate_maintenance": tool_estimate_maintenance,
    "detect_plateau": tool_detect_plateau,
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


def run_tool(name, arguments):
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        return function(**(arguments or {}))
    except TypeError:
        return {"error": "Invalid arguments for this tool."}
    except Exception as error:
        return {"error": f"Tool failed: {type(error).__name__}"}