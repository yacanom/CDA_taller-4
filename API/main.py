from typing import List

import pandas as pd

from fastapi import FastAPI

from model import DataModel
from prediction_model import PredictionModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
   return { "message": "Hello world" }

@app.post("/retraining")
def test():
    return { "message": "Re entreno"}

@app.post("/predict")
def make_predictions(X: List[DataModel]):
    df = pd.DataFrame([x.dict() for x in X])
    predicion_model = PredictionModel()
    results = predicion_model.make_predictions(df)
    return results.tolist()

@app.get("/version")
def versions():
    return {"message": "Version"}