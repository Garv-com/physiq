def get_session_top_weights(df, exercise):
    exercise_df = df[df["exercise"] == exercise]
    session_summary = exercise_df.groupby(["session_id", "date"])["weight_kg"].max().reset_index()
    session_summary = session_summary.sort_values("date").reset_index(drop=True)
    return session_summary