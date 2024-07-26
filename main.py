import uvicorn
from fastapi import FastAPI
import pickle
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
    
# Run the app
# uvicorn main:app --reload