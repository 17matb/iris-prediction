from fastapi import FastAPI
from pydantic import BaseModel


class Prediction(BaseModel):
    first_value: float


app = FastAPI()


@app.post("/predict")
def predict(input: Prediction):
    prediction = input.first_value * 2
    return {"prediction": prediction}
