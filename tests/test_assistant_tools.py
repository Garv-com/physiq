import json

import src.assistant_tools as tools
from tests.helpers import seed_nutrition


def test_unknown_tool_returns_an_error(temp_db):
    assert "error" in tools.run_tool("does_not_exist", {})


def test_bad_arguments_return_an_error_instead_of_raising(temp_db):
    assert "error" in tools.run_tool("predict_weight", {"unexpected": 1})


def test_empty_database_gives_errors_not_crashes(temp_db):
    for name in ["get_recent_weight", "predict_weight", "estimate_maintenance", "detect_plateau", "detect_anomalies"]:
        result = tools.run_tool(name, {})
        assert "error" in result, name
        assert "units" not in result


def test_successful_results_carry_units_and_are_json_serialisable(temp_db):
    seed_nutrition(days=60)
    for name, args in [
        ("get_recent_weight", {"days": 5}),
        ("predict_weight", {"days": 7}),
        ("estimate_maintenance", {}),
        ("detect_plateau", {}),
        ("detect_anomalies", {"days": 30}),
        ("get_model_explanation", {}),
    ]:
        result = tools.run_tool(name, args)
        assert "error" not in result, (name, result)
        assert "units" in result
        json.dumps(result)


def test_non_numeric_day_counts_fall_back_to_defaults(temp_db):
    seed_nutrition(days=60)
    result = tools.run_tool("get_recent_weight", {"days": "abc"})
    assert result["days_requested"] == 14


def test_day_counts_are_clamped(temp_db):
    seed_nutrition(days=60)
    assert tools.run_tool("get_recent_weight", {"days": 100000})["days_requested"] == 90


def test_anomaly_tool_reports_a_spike_inside_the_window(temp_db):
    data = seed_nutrition(days=60)
    spike_date = data["date"].iloc[-3].date().isoformat()
    temp_db.add_nutrition_entry(spike_date, float(data["calories"].iloc[-3]), float(data["weight"].iloc[-3]) + 3.0)
    result = tools.run_tool("detect_anomalies", {"days": 10})
    assert spike_date in [item["date"] for item in result["anomalies"]]


def test_analyze_progress_rejects_unknown_exercises(temp_db):
    temp_db.add_workout_set("2026-03-01", "Squat", 100.0, 5, None)
    assert "error" in tools.run_tool("analyze_progress", {"exercise": "Curls"})


def test_every_tool_schema_has_a_matching_function():
    names = {schema["name"] for schema in tools.TOOL_SCHEMAS}
    assert names == set(tools.TOOL_FUNCTIONS)