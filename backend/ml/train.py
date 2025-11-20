from pathlib import Path

import joblib
import mlflow
from azureml.core import Workspace
from dotenv import load_dotenv
from logs.logger import get_logger
from mlflow import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "model.pkl"
logger.info(f"Model will be saved in {MODEL_PATH}")

ws = Workspace.from_config()
tracking_uri = ws.get_mlflow_tracking_uri()

logger.info("Using Azure ML Tracking Server")
mlflow.set_tracking_uri(tracking_uri)

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

sklearn.autolog()

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(lr, MODEL_PATH)
logger.info(f"Successfully saved model in {MODEL_PATH}")
