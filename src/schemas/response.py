from pydantic import BaseModel
from typing import Optional, List, Dict

class PredictResponse(BaseModel):
    label: str
    confidence: float
    explanation: str
    extracted_title: Optional[str] = None
    rag_analysis: Optional[str] = None
    retrieved_sources: Optional[List[Dict]] = None
