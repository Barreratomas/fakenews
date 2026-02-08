from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class PredictResponse(BaseModel):
    label: str
    confidence: float
    explanation: str
    extracted_title: Optional[str] = None
    rag_analysis: Optional[str] = None
    retrieved_sources: Optional[List[Dict]] = None
    model_explanation: Optional[Dict] = None
    text: Optional[str] = None
    conflict_resolution: Optional[Dict[str, Any]] = None
