import os
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, recall_score, accuracy_score, precision_score

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task
def train_model(X_train, y_train):
    best_params = {
        "n_estimators": 22,
        "max_depth": 11,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)
    print("training is complete")
    return rf


@task
def save_model(model, dv):
    '''
    Saves the new version of a model
    '''
    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    p = parent_dir + r'/src/prediction_service/model.bin'
    print(p)
    pickle.dump((dv, model), open(p, 'wb'))


def apply_standard_scaling(df):
    ss = StandardScaler(with_mean=False)
    df_scaled = ss.fit_transform(df)
    return df_scaled


@task(name='Performance Metrics', retries=3)
def calculate_metrics(model, X_val, y_val):
    '''
    Calculates the score metrics for a test data
    '''
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    metrics_dict = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }
    return metrics_dict


@task
def save_test_dataset(df, test_target):
    '''
    Rewrites test dataset in ./evidently_service/datasets/
    '''
    X_test = pd.DataFrame(df.toarray())
    y_test = pd.DataFrame(test_target)
    test_df = X_test.loc[y_test.index]
    parent_dir = os.path.dirname(os.path.abspath("__file__"))
    p = parent_dir + r"/src/evidently_service/datasets/test.csv"
    print(p)
    test_df.to_csv(p, index=False)
    print("dataset is saved")


# mlflow.sklearn.autolog()

# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("france-renewable-production")

@flow(task_runner=SequentialTaskRunner())
def run_train():
    data_path = "./output"
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        mlflow.log_param("raw-data-path", "./data/intermittent-renewables-production-france.csv")

        # Create tags and log params
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = "renewable-prediction-model"
        mlflow.set_tag("developer", "shreyas")

        mlflow.set_tag("model_type", "logistic_regression")
        X_train_scaled = apply_standard_scaling(X_train)
        model = train_model(X_train_scaled, y_train)

        # metrics calculation
        metrics_dict = calculate_metrics(model, X_val, y_val)
        mlflow.log_metrics(metrics_dict)

        dv = DictVectorizer()
        save_model(model, dv)
        print("Model has been trained and saved")

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        mlflow.log_artifacts(local_dir="artifacts")
        mlflow.register_model(model_uri=model_uri, name=model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Production",
            archive_existing_versions=False,
        )
        save_test_dataset(X_test, y_test)


if __name__ == '__main__':
    run_train()
