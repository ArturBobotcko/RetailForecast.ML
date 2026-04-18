from pydantic import BaseModel

from models.Model import Model


class TrainingRunRequest(BaseModel):
    trainingRunId: int
    datasetId: int
    downloadUrl: str
    callbackUrl: str
    timeColumn: str = "Год"
    forecastHorizon: int = 1
    forecastFrequency: str = "Auto"
    targetColumn: str
    featureColumns: list[str]
    model: Model
