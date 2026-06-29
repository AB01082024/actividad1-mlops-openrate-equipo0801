import pandas as pd
import yaml
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train(params_path):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    train_path = params["split"]["train_path"]
    test_path = params["split"]["test_path"]
    seed = params["model"]["random_state"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    experiment_name = params["mlflow"]["experiment_name"]
    tracking_uri = params["mlflow"]["tracking_uri"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    features = ["hour_of_day", "day_of_week", "historical_open_rate",
                "historical_push_count", "days_since_last_open"]
    target = "target_opened"

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_params({
            "model": "random_forest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": seed
        })
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "auc": auc})
        mlflow.sklearn.log_model(model, "random_forest_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/random_forest_model.joblib")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    train(args.params)
    