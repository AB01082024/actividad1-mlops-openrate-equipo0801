import pandas as pd
import numpy as np
import yaml
import argparse
import os

def generate_data(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    n = params["data"]["n_samples"]
    seed = params["data"]["random_state"]
    output_path = params["data"]["output_path"]

    np.random.seed(seed)

    df = pd.DataFrame({
        "user_id": [f"U{str(i).zfill(5)}" for i in range(n)],
        "site": np.random.choice(["sitioA", "sitioB", "sitioC"], n),
        "campaign_type": np.random.choice(["promo", "info", "alerta"], n),
        "device_os": np.random.choice(["android", "ios", "windows"], n),
        "hour_of_day": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "historical_open_rate": np.round(np.random.uniform(0, 1, n), 2),
        "historical_push_count": np.random.randint(0, 100, n),
        "days_since_last_open": np.random.randint(0, 30, n),
        "segment": np.random.choice(["A", "B", "C"], n),
    })

    df["target_opened"] = (
        (df["historical_open_rate"] > 0.5) &
        (df["days_since_last_open"] < 10)
    ).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Datos generados: {output_path} ({n} registros)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    generate_data(args.params)
