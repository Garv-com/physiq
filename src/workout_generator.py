import numpy as np
import pandas as pd


def generate_workout_data(
    weeks=12,
    sessions_per_week=4,
    exercises=("Squat", "Bench Press", "Deadlift", "Overhead Press"),
    start_weight=60.0,
    weekly_progress_kg=0.5,
    reps_mean=8,
    reps_std=1.5,
    rpe_mean=7.5,
    rpe_std=1.0,
    weight_noise_std=2.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    records = []
    session_id = 0

    for week in range(weeks):
        for session_num in range(sessions_per_week):
            session_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=week * 7 + session_num * 2)
            exercise = exercises[session_id % len(exercises)]

            true_top_weight = start_weight + weekly_progress_kg * week
            num_sets = rng.integers(3, 5)

            for set_num in range(1, num_sets + 1):
                observed_weight = true_top_weight + rng.normal(0, weight_noise_std)
                reps = max(1, round(rng.normal(reps_mean, reps_std)))
                rpe = np.clip(rng.normal(rpe_mean, rpe_std), 5, 10)

                records.append({
                    "session_id": session_id,
                    "date": session_date,
                    "exercise": exercise,
                    "set_number": set_num,
                    "weight_kg": round(observed_weight, 1),
                    "reps": reps,
                    "rpe": round(rpe, 1),
                    "true_top_weight": round(true_top_weight, 2),
                })

            session_id += 1

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_workout_data()
    df.to_csv("data/raw/synthetic_workout_log.csv", index=False)
    print(df.head(15))
    print(f"\nGenerated {df['session_id'].nunique()} sessions, {len(df)} total sets.")