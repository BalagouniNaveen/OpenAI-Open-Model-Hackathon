from pydantic import BaseModel
from typing import Optional

class SensorPayload(BaseModel):
    sensor_id: str
    type: str
    value: float
    timestamp: Optional[str] = None

class EventRecord(BaseModel):
    id: int
    sensor_id: str
    type: str
    value: float
    timestamp: Optional[str] = None
    is_anomaly: bool
    ml_score: float
    gpt_reason: Optional[str] = None
    rpa_action: Optional[str] = None
