import pandas as pd


def get_session_top_weights(df, exercise):
    exercise_df = df[df["exercise"] == exercise]
    session_summary = exercise_df.groupby(["session_id", "date"])["weight_kg"].max().reset_index()
    session_summary = session_summary.sort_values("date").reset_index(drop=True)
    return session_summary


def add_estimated_1rm(df, weight_col="weight_kg", reps_col="reps", output_col="e1rm"):
    result = df.copy()
    result[output_col] = result[weight_col] * (1 + result[reps_col] / 30)
    result[output_col] = result[output_col].round(1)
    return result


def detect_prs(df, exercise, weight_col="weight_kg", reps_col="reps", use_e1rm=True):
    exercise_df = df[df["exercise"] == exercise].copy()
    exercise_df = exercise_df.sort_values("date")

    if use_e1rm:
        exercise_df = add_estimated_1rm(exercise_df, weight_col=weight_col, reps_col=reps_col)
        compare_col = "e1rm"
    else:
        compare_col = weight_col

    running_max = 0
    is_pr = []

    for value in exercise_df[compare_col]:
        if value > running_max:
            is_pr.append(True)
            running_max = value
        else:
            is_pr.append(False)

    exercise_df["is_pr"] = is_pr
    return exercise_df[exercise_df["is_pr"]]


def get_session_volume(df, weight_col="weight_kg", reps_col="reps"):
    volume_df = df.copy()
    volume_df["set_volume"] = volume_df[weight_col] * volume_df[reps_col]

    session_volume = volume_df.groupby(["session_id", "date", "exercise"])["set_volume"].sum().reset_index()
    session_volume = session_volume.rename(columns={"set_volume": "total_volume"})

    return session_volume.sort_values("date").reset_index(drop=True)