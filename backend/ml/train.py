import os
from pathlib import Path

import joblib
import mlflow
from dotenv import load_dotenv
from logs.logger import get_logger
from mlflow import exceptions, sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "model.pkl"
logger.info(f"Model will be saved in {MODEL_PATH}")
DB_PATH = BASE_DIR / "mlruns.db"
logger.info(f"Database will be saved in {DB_PATH}")
ARTIFACT_PATH = BASE_DIR / "mlruns"
logger.info(f"Artifact directory will be at {ARTIFACT_PATH}")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")

if MLFLOW_TRACKING_URI and os.getenv("GITHUB_ACTIONS") == "true":
    logger.info("Using Azure ML Tracking Server")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    logger.info(f"Using local SQLite tracking: sqlite:///{DB_PATH}")
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

experiment_name = "MLFlow quickstart"

if MLFLOW_TRACKING_URI and os.getenv("GITHUB_ACTIONS") == "true":
    try:
        mlflow.create_experiment(experiment_name)
    except exceptions.MlflowException:
        pass
else:
    try:
        mlflow.create_experiment(
            experiment_name, artifact_location=ARTIFACT_PATH.as_uri()
        )
    except exceptions.MlflowException:
        pass


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
