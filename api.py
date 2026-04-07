# Building Inference API using FastAPI and Pickle
# from fastapi import FastAPI
# import pickle
# import numpy as np

# app = FastAPI()

# model = pickle.load(open("C:/Users/User/ai-ml-career/ml-pipeline-engineering/model.pkl", "rb"))

# @app.get("/")
# def home():
#     return {"message": "ML Model API is initialized and ready to receive requests."}

# @app.post("/predict")
# def predict(data: list):
#     arr = np.array(data).reshape(1, -1)
#     pred = model.predict(arr)
#     return {"prediction": pred.tolist()} 

# what was wrong in this code snippet is that the input data for the predict function was not properly defined.
# The need for class InputData(BaseModel) is to create a schema for the input data that the predict function
# will receive.
# what app.get("/") does is it defines a GET endpoint at the root URL ("/") of the API. 
# When a client sends a GET request to this endpoint, the home function which is 'predict' in this case
# will be executed, and it will return a JSON response with a message indicating that the ML Model API is 
# initialized and ready to receive requests. 

# GET vs POST: GET requests are typically used to retrieve data from the server,
# while POST requests are used to send data to the server for processing. 
# In this case, we use POST for the /predict endpoint because we are sending input data to the server for prediction.
# The predict function takes the input data, processes it, and returns the prediction result as a 
# JSON response. What we would have to do in case of GET request is to pass the input data as query parameters in the URL,
# like /predict?data=17.9,10.3,122.8,1001,0.1184,... which is not ideal for complex data like feature arrays.
# this code line /predict?data=17.9,10.3,122.8,1001,0.1184,... will come after the application startup is complete
# and we want to test the API using a GET request.
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
app = FastAPI()
model = pickle.load(open("C:/Users/User/ai-ml-career/ml-pipeline-engineering/model.pkl", "rb"))

class InputData(BaseModel): # Added input schema for better validation using Pydantic's BaseModel.
    data : list             # This tells FastAPI to expect a list of features in the request body
@app.post("/predict")       
def predict(input_data: InputData):
    arr = np.array(input_data.data).reshape(1,-1)
    pred = model.predict(arr)
    return {"prediction": pred.tolist()}


# app.post decorator to define the endpoint for prediction. what it actually does is it tells 
# FastAPI that this function should be called when a POST request is made to the /predict endpoint.
# def predict(input_data: InputData): This is the function that will be called when a POST request 
# is made to the /predict endpoint. It takes an argument input_data of type InputData,
# which is a Pydantic model that we defined to validate the incoming data.
# arr = np.array(input_data.data).reshape(1,-1): This line converts the input data into a NumPy array 
# and reshapes it to ensure it has the correct dimensions for prediction.
# pred = model.predict(arr): This line uses the loaded machine learning model to make a prediction
# return {"prediction": pred.tolist()}: This line returns the prediction as a JSON response. 
# The pred.tolist() converts the NumPy array to a regular list, which can be easily serialized to JSON.

# "uvicorn api:app --reload"  --use this command(without the quotes) to run the API server
# Make sure you are in the directory where api.py is located and that you have uvicorn installed. 
# The --reload flag allows the server to automatically reload when you make changes to the code,
# which is useful during development.


# once application startup is complete go to http://127.0.0.1:8000/docs to 
# access the interactive API documentation provided by FastAPI.

# Example of input data for prediction in POST request to /predict endpoint:
# {
#   "data": [
# 17.9, 10.3, 122.8, 1001, 0.1184,
# 0.2776, 0.3001, 0.1471, 0.2419,
# 0.07871, 1.095, 0.9053, 8.589,
# 153.4, 0.006399, 0.04904, 0.05373,
# 0.01587, 0.03003, 0.006193,
# 25.83, 17.33, 184.6, 2019,
# 0.1622, 0.6656, 0.7119, 0.2654,
# 0.4601,0.1189
#   ]
# }

# Output will be something like this:
	
# Response body
# {
#   "prediction": [
#     0
#   ]
# }
# What this means is that the model has predicted the class label for the input data,
# and in this case, it has predicted class 0, means cancer is detected.

