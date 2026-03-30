from pydantic import BaseModel
from models.Model import Model

class TrainingRunRequest(BaseModel):
    trainingRunId: int
    datasetId: int
    downloadUrl: str
    callbackUrl: str
    timeColumn: str = "Год"
    forecastPeriod: int | None = None
    targetColumn: str
    featureColumns: list[str]
    model: Model
