from pydantic import BaseModel
from models.Metric import Metric

class TrainingRunCallbackRequest(BaseModel):
    status: str
    metrics: list[Metric] = []
    forecast: list[dict[str, object]] = []
    error: str | None = None
    externalJobId: str | None = None
