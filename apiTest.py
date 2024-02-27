import requests
import uvicorn
from fastapi import FastAPI
from typing import Union, Any, List
from pydantic import BaseModel
from onnxInference import *

app = FastAPI()

model_path = "./models/dy_model_new.onnx"
config_name = "./config/标签类别映射/config72.json"

def predict(input_data: List[str]) -> List[str]:
    label_list, pred_probs = main_inferance(input_data, config_name, model_path)
    return [{"label": label, "probability": pred} for label, pred in zip(label_list, pred_probs)]

class PredictionRequest(BaseModel):
    data: List[str]

class PredictionResponse(BaseModel):
    predictions: List[Any]



@app.post("/predict/", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    predictions = predict(request.data)
    return PredictionResponse(predictions=predictions)


if __name__ == "__main__":
    
    uvicorn.run(app=app, host="127.0.0.1", port=9000, reload=True)
    


