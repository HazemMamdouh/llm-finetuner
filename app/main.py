from fastapi import FastAPI
from typing import  List
from app.api import api
from app.modules.data_model import InputData, OutputData

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/alive")
def alive_check():
    """
    define endpoint to check the server aliveness

    """
    return {"message": "Server is Alive"}


@app.get("/models/{model_id}/status")
def check_model_status(model_id: str):
    """
    define endpoint to get the status of the pre-requested model id

    """
    return api.model_event_status(model_id)


@app.post("/models/{model_id}/train", response_model=OutputData)
def finetune_model (payload: List[InputData], model_id: str):
    """
    define endpoint to initiate the fine tuning of the LLM with the given data

    """
    response = api.finetune_model(payload, model_id)

    return response
