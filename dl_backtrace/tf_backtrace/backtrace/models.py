from typing import List, Dict, Optional
from pydantic import BaseModel

class InitModel(BaseModel):
    model_link: str
    model_type: str
    token: str

class DelModel(BaseModel):
    token: str

class Datapoint(BaseModel):
    data: list
    is_multi_input: bool
    mode: str
    token: str

# class Sample(BaseModel):
#     model_code: str
#     data: List[Datapoint]
#     structure: str

# class BTresponse(BaseModel):
#     relevance: dict
#     status: str