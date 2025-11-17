import joblib
import mlflow
from mlflow import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("MLflow Quickstart")

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

sklearn.autolog()

with mlflow.start_run():
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    joblib.dump(lr, "../model/model.pkl")
