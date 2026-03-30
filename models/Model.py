from pydantic import BaseModel

class Model(BaseModel):
    id: int
    name: str
    algorithm: str