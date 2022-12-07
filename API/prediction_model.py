from joblib import load
from sklearn.preprocessing import PowerTransformer

class PredictionModel:

    def __init__(self):
        self.model = load("../model/modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
         
    def make_proba_predictions(self, data):
        result = self.model.predict_proba(data)
        return result

    def make_predictions2(self, data):
        self.model2 = load("../model/modelo_nuevo.joblib")
        result = self.model2.predict(data)
        return result
    
    def make_fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        #save model
       
        from joblib import dump
        dump(self.model, "../model/modelo_nuevo.joblib")
        
        
        return {"message": "Modelo reentrenado"} 
        
        
