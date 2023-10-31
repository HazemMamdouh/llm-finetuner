from pydantic import BaseModel



class InputData(BaseModel):
    instruction: str
    input: str
    output: str


class OutputData(BaseModel):
    message: str
    gpu_id: int
    gpu_type: str
    model_id: str
