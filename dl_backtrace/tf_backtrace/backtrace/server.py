from fastapi import FastAPI
from models import *
from utils import load_and_store, compute, delete_model

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Operational"}


@app.post("/init")
async def init_model(init_model: InitModel):
    resp = load_and_store(init_model)
    return resp

@app.post("/eval")
async def process(sample: Datapoint):
    resp = compute(sample)
    return resp

@app.post("/delete")
async def remove_model(del_model: DelModel):
    resp = delete_model(del_model)
    return resp

# uvicorn server:app --reload