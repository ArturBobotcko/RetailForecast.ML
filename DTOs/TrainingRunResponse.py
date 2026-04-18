from pydantic import BaseModel


class TrainingRunResponse(BaseModel):
    externalJobId: str
    status: str
    message: str
