import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get("/")
async def index():
    return {"message": "Hello World"}

@app.get("/{name}")
async def get_name(name:str):
    return {"message": f"Hello, {name}"}

@app.post("/predict")
async def predict(data: BankNote):
    print(data)
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    if prediction[0] > 0.5:
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
# Run the app
# uvicorn main:app --reload