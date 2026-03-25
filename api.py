from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("C:/Users/User/ai-ml-career/ml-pipeline-engineering/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML Model API is initialized and ready to receive requests."}

@app.post("/predict")
def predict(data: list):
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": pred.tolist()}