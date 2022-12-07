from typing import List

import pandas as pd

from sklearn.metrics import recall_score

from fastapi import FastAPI, File, UploadFile


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


@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    df_original = pd.read_json('https://raw.githubusercontent.com/yacanom/CDA_taller-4/main/DataSet_Entrenamiento_v1.json')
    
    df_original['TotalCharges'] =  pd.to_numeric(df_original['TotalCharges'])
    df_original['TotalCharges'] = df_original['TotalCharges'].fillna(df_original['TotalCharges'].median())
    x_trainNew = df_original.drop(['customerID', 'Churn'],axis=1)
    predicion_model = PredictionModel()
    results = predicion_model.make_predictions(x_trainNew)
    report_train = recall_score(df_original['Churn'], results, average='macro')
    #---------------------------------------------------------------
    df = pd.read_json(file.file)
    df['TotalCharges'] =  pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    x_trainNew = df.drop(['customerID', 'Churn'],axis=1)
    y_trainNew = df['Churn']
    fit_model = PredictionModel()
    results = fit_model.make_fit(x_trainNew, y_trainNew)
    
    predicion_model2 = PredictionModel()
    results = predicion_model2.make_predictions2(x_trainNew)
    report_train2 = recall_score(df['Churn'], results, average='macro')
    
    return {"Recall: original":report_train, "Recall: nuevo":report_train2}
    

    
@app.post("/predict2")
def make_massive_predictions(file: UploadFile = File(...)):
    df = pd.read_json(file.file)
    df['TotalCharges'] =  pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    x_trainNew = df.drop(['customerID', 'Churn'],axis=1)
    predicion_model = PredictionModel()
    proba = predicion_model.make_proba_predictions(x_trainNew)
    #test_preds_proba = best_model.predict_proba(x_test)[:, 1]
    #test_preds_proba = predicion_model.predict_proba(x_test)
    predictions = predicion_model.make_predictions(x_trainNew)
    predictions = pd.DataFrame(predictions, columns=['predictions'])
    proba = pd.DataFrame(proba, columns=['Yes', 'No'])
    predictions = pd.concat([predictions, proba], axis=1)
    
    
    
    #report = recall_score(df['Churn'], results, average='macro')
    return {"predictions":predictions.to_dict(orient='records')}

@app.post("/predict")
def make_predictions(X: List[DataModel]):
    df = pd.DataFrame([x.dict() for x in X])
    predicion_model = PredictionModel()
    results = predicion_model.make_predictions(df)
    return results.tolist()

@app.post("/version")
def versions(X: List[DataModel], version: int):
    df = pd.DataFrame([x.dict() for x in X])
    predicion_model = PredictionModel()
    if version == 1:
        results = predicion_model.make_predictions(df)
    elif version == 2:
        results = predicion_model.make_predictions2(df)
    return results.tolist()
