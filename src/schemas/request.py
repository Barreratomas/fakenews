from pydantic import BaseModel, HttpUrl
from typing import Literal

class PredictRequest(BaseModel):
    type: Literal["text", "url"]
    content: str
