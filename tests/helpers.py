import src.database as database
from src.synthetic_generator import generate_weight_data


def seed_nutrition(days=60, seed=5, **kwargs):
    data = generate_weight_data(days=days, seed=seed, **kwargs)
    for _, row in data.iterrows():
        database.add_nutrition_entry(
            row["date"].date().isoformat(), float(row["calories"]), float(row["weight"])
        )
    return data