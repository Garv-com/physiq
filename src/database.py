import sqlite3
from contextlib import contextmanager

import pandas as pd

DB_PATH = "data/physiq.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nutrition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                calories REAL NOT NULL,
                weight REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                exercise TEXT NOT NULL,
                set_number INTEGER NOT NULL,
                weight_kg REAL NOT NULL,
                reps INTEGER NOT NULL,
                rpe REAL
            )
            """
        )


def seed_from_csv(nutrition_csv, workout_csv):
    nutrition = pd.read_csv(nutrition_csv)[["date", "calories", "weight"]]
    workouts = pd.read_csv(workout_csv)[
        ["session_id", "date", "exercise", "set_number", "weight_kg", "reps", "rpe"]
    ]

    with get_connection() as conn:
        if conn.execute("SELECT COUNT(*) FROM nutrition_logs").fetchone()[0] == 0:
            nutrition.to_sql("nutrition_logs", conn, if_exists="append", index=False)
        if conn.execute("SELECT COUNT(*) FROM workout_sets").fetchone()[0] == 0:
            workouts.to_sql("workout_sets", conn, if_exists="append", index=False)


def load_nutrition_df():
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT date, calories, weight FROM nutrition_logs ORDER BY date", conn
        )


def load_workout_df():
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT session_id, date, exercise, set_number, weight_kg, reps, rpe "
            "FROM workout_sets ORDER BY date, session_id, set_number",
            conn,
        )

def add_nutrition_entry(date, calories, weight):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO nutrition_logs (date, calories, weight) VALUES (?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET calories = excluded.calories, weight = excluded.weight",
            (date, calories, weight),
        )


def add_workout_set(date, exercise, weight_kg, reps, rpe):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT session_id FROM workout_sets WHERE date = ? LIMIT 1", (date,)
        ).fetchone()

        if row:
            session_id = row[0]
        else:
            session_id = conn.execute(
                "SELECT COALESCE(MAX(session_id), -1) + 1 FROM workout_sets"
            ).fetchone()[0]

        set_number = conn.execute(
            "SELECT COUNT(*) FROM workout_sets WHERE session_id = ? AND exercise = ?",
            (session_id, exercise),
        ).fetchone()[0] + 1

        conn.execute(
            "INSERT INTO workout_sets (session_id, date, exercise, set_number, weight_kg, reps, rpe) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, date, exercise, set_number, weight_kg, reps, rpe),
        )

    return {"session_id": session_id, "set_number": set_number}

if __name__ == "__main__":
    init_db()
    seed_from_csv(
        "data/raw/synthetic_nutrition_log.csv",
        "data/raw/synthetic_workout_log.csv",
    )
    print(f"Nutrition rows: {len(load_nutrition_df())}")
    print(f"Workout set rows: {len(load_workout_df())}")

def load_recent_nutrition(days):
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT date, calories, weight FROM nutrition_logs "
            "WHERE date >= date((SELECT MAX(date) FROM nutrition_logs), ?) ORDER BY date",
            conn,
            params=(f"-{days - 1} days",),
        )