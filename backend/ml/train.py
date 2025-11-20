from pathlib import Path

import joblib
import mlflow
from azureml.core import Model, Workspace
from dotenv import load_dotenv
from logs.logger import get_logger
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "model.pkl"

ws = Workspace.from_config()
tracking_uri = ws.get_mlflow_tracking_uri()

mlflow.set_tracking_uri(tracking_uri)
logger.info("Using Azure ML Tracking Server")

experiment_name = "MLFlow quickstart"

mlflow.create_experiment(experiment_name)


mlflow.set_experiment(experiment_name)

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

with mlflow.start_run():
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", lr.score(X_test, y_test))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, MODEL_PATH)
    Model.register(
        workspace=ws,
        model_path=str(MODEL_PATH),
        model_name="iris-logistic-regression",
        description="Logistic regression model for iris prediction",
    )
