import pandas as pd
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split

def prepare_data(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    input_path = params["data"]["output_path"]
    test_size = params["split"]["test_size"]
    seed = params["split"]["random_state"]
    train_path = params["split"]["train_path"]
    test_path = params["split"]["test_path"]

    df = pd.read_csv(input_path)

    train, test = train_test_split(df, test_size=test_size, random_state=seed)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Train: {len(train)} registros → {train_path}")
    print(f"Test: {len(test)} registros → {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    prepare_data(args.params)