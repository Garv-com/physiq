from tests.helpers import seed_nutrition


def test_logging_the_same_day_twice_updates_instead_of_duplicating(temp_db):
    temp_db.add_nutrition_entry("2026-03-01", 2400.0, 75.0)
    temp_db.add_nutrition_entry("2026-03-01", 2500.0, 75.4)
    df = temp_db.load_nutrition_df()
    assert len(df) == 1
    assert df["calories"].iloc[0] == 2500.0
    assert df["weight"].iloc[0] == 75.4


def test_sets_on_one_day_share_a_session_and_count_up(temp_db):
    first = temp_db.add_workout_set("2026-03-01", "Squat", 100.0, 5, 8.0)
    second = temp_db.add_workout_set("2026-03-01", "Squat", 100.0, 5, 8.5)
    other_exercise = temp_db.add_workout_set("2026-03-01", "Bench Press", 60.0, 8, None)

    assert first["session_id"] == second["session_id"] == other_exercise["session_id"]
    assert (first["set_number"], second["set_number"]) == (1, 2)
    assert other_exercise["set_number"] == 1


def test_a_new_day_starts_a_new_session(temp_db):
    first = temp_db.add_workout_set("2026-03-01", "Squat", 100.0, 5, None)
    later = temp_db.add_workout_set("2026-03-08", "Squat", 102.5, 5, None)
    assert later["session_id"] == first["session_id"] + 1
    assert later["set_number"] == 1


def test_text_with_sql_characters_is_stored_literally(temp_db):
    nasty = "Curl'; DROP TABLE workout_sets; --"
    temp_db.add_workout_set("2026-03-01", nasty, 20.0, 10, None)
    df = temp_db.load_workout_df()
    assert df["exercise"].iloc[0] == nasty
    assert len(df) == 1


def test_recent_window_is_measured_back_from_the_latest_entry(temp_db):
    seed_nutrition(days=30)
    recent = temp_db.load_recent_nutrition(7)
    everything = temp_db.load_nutrition_df()
    assert len(recent) == 7
    assert recent["date"].iloc[-1] == everything["date"].iloc[-1]


def test_empty_database_returns_empty_frames(temp_db):
    assert temp_db.load_nutrition_df().empty
    assert temp_db.load_workout_df().empty
    assert temp_db.load_recent_nutrition(30).empty