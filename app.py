import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = pickle.load(open("model.pkl", "rb"))

class Furniture(BaseModel):
    category: int = Field(ge=0, le=16)
    sellable_online: int = Field(ge=0, le=1)
    other_colors: int = Field(ge=0, le=1)
    depth: float = Field(gt=0)
    height: float = Field(gt=0)
    width: float = Field(gt=0)

def predict_price(features: Furniture) -> float:
    X = pd.DataFrame([features.model_dump()])
    return round(float(model.predict(X)[0]), 2)

# ---------- HTML interface (like Flask) ----------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            category: int = Form(...), sellable_online: int = Form(...),
            other_colors: int = Form(...), depth: float = Form(...),
            height: float = Form(...), width: float = Form(...)):
    try:
        features = Furniture(category=category, sellable_online=sellable_online,
                             other_colors=other_colors, depth=depth,
                             height=height, width=width)
    except ValidationError as e:
        errors = ", ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return templates.TemplateResponse(
            request, "index.html", {"prediction_text": f"Invalid input. {errors}"},
            status_code=422)
    price = predict_price(features)
    return templates.TemplateResponse(
        request, "index.html",
        {"prediction_text": f"Furniture prediction price is : $ {price}"})

# ---------- JSON API (Swagger: /docs) ----------
@app.post("/make_predictions")
async def make_predictions(features: Furniture):
    return {"predicted_price": predict_price(features)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
