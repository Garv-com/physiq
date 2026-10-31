import pandas as pd
import pytest

from src.training import (
    add_estimated_1rm,
    build_exercise_summary,
    detect_prs,
    get_session_top_weights,
    get_session_volume,
)

COLUMNS = ["session_id", "date", "exercise", "set_number", "weight_kg", "reps", "rpe"]


def sets_frame():
    rows = [
        (0, "2026-01-01", "Squat", 1, 100.0, 5, 8.0),
        (0, "2026-01-01", "Squat", 2, 90.0, 8, 8.0),
        (1, "2026-01-02", "Bench Press", 1, 60.0, 5, 8.0),
        (2, "2026-01-08", "Squat", 1, 105.0, 5, 8.0),
        (2, "2026-01-08", "Squat", 2, 95.0, 8, 8.0),
        (3, "2026-01-15", "Squat", 1, 100.0, 5, 8.0),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_estimated_1rm_uses_the_epley_formula():
    result = add_estimated_1rm(sets_frame())
    assert result["e1rm"].iloc[0] == pytest.approx(100.0 * (1 + 5 / 30), abs=0.05)


def test_session_volume_is_weight_times_reps_summed():
    volume = get_session_volume(sets_frame()[lambda d: d["exercise"] == "Squat"])
    assert volume.loc[volume["session_id"] == 0, "total_volume"].iloc[0] == 100 * 5 + 90 * 8


def test_top_weight_is_the_heaviest_set_per_session():
    tops = get_session_top_weights(sets_frame(), "Squat")
    assert tops["weight_kg"].tolist() == [100.0, 105.0, 100.0]


def test_prs_are_new_best_estimated_1rms():
    prs = detect_prs(sets_frame(), "Squat")
    assert prs["session_id"].tolist() == [0, 2]


def test_summary_flags_pr_sessions_and_only_includes_the_exercise():
    summary = build_exercise_summary(sets_frame(), "Squat")
    assert summary["is_pr_session"].tolist() == [True, True, False]
    assert len(summary) == 3
    assert {"weight_kg", "trend_top_weight", "total_volume", "is_pr_session"} <= set(summary.columns)