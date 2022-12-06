from joblib import load
from sklearn.preprocessing import PowerTransformer

class PredictionModel:

    def __init__(self):
        self.model = load("/RF.joblib")
        self.pt = load("/pt.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
