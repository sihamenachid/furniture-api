import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Furniture price prediction API")

# The trained model, loaded once when the API starts
model = pickle.load(open("model.pkl", "rb"))


class Furniture(BaseModel):
    category: int = Field(ge=0, le=16)
    sellable_online: int = Field(ge=0, le=1)
    other_colors: int = Field(ge=0, le=1)
    depth: float = Field(gt=0)
    height: float = Field(gt=0)
    width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "ML model for furniture price prediction. Go to /docs to test it."}


@app.post("/make_predictions")
async def make_predictions(features: Furniture):
    X = pd.DataFrame([features.model_dump()])
    price = model.predict(X)[0]
    return {"predicted_price": round(float(price), 2)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)