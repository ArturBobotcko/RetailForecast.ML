from fastapi import FastAPI

from app.api.training_runs import router as training_runs_router
from app.config import set_all_seeds


def create_app() -> FastAPI:
    set_all_seeds()
    app = FastAPI()
    app.include_router(training_runs_router)
    return app


app = create_app()
