from fastapi import FastAPI

from app.api import api

app = FastAPI()


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
def finetune_model (payload: InputData[], model_id: str):
    """
    define endpoint to initiate the fine tuning of the LLM with the given data

    """
    payload = payload.dict()

    return api.start_finetuning(payload, model_id)
