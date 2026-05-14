from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks

from app.application.training_runs import process_training_run
from DTOs.TrainingRunRequest import TrainingRunRequest
from DTOs.TrainingRunResponse import TrainingRunResponse

router = APIRouter()


@router.post("/api/trainingrun/start", response_model=TrainingRunResponse)
async def training_run_start(
    request: TrainingRunRequest, background_tasks: BackgroundTasks
) -> TrainingRunResponse:
    external_job_id = str(uuid4())
    background_tasks.add_task(process_training_run, request, external_job_id)

    return TrainingRunResponse(
        externalJobId=external_job_id,
        status="Running",
        message=f"Training started for run {request.trainingRunId}",
    )
