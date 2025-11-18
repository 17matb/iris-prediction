import joblib
from fastapi import FastAPI
from logs.logger import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)
app = FastAPI()


logger.info("Loading model...")
model = joblib.load("./model/model.pkl")
logger.info("Successfully loaded model")


class Iris(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float


@app.post("/predict")
def predict(input: Iris):
    X = [
        [
            input.sepal_length_cm,
            input.sepal_width_cm,
            input.petal_length_cm,
            input.petal_width_cm,
        ]
    ]
    y_pred = model.predict(X)
    return {"prediction": int(y_pred[0])}
