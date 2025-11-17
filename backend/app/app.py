import joblib
from fastapi import FastAPI
from pydantic import BaseModel


class Iris(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float


app = FastAPI()

model = joblib.load("./model/model.pkl")


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
