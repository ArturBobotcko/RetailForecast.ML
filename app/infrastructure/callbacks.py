import httpx

from DTOs.TrainingRunCallbackRequest import TrainingRunCallbackRequest


async def send_callback(callback_url: str, payload: TrainingRunCallbackRequest) -> None:
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        response = await client.post(callback_url, json=payload.model_dump(mode="json"))
        if response.is_success:
            return

        response_body = response.text.strip()
        if response_body:
            raise RuntimeError(
                f"Callback failed with status {response.status_code}: {response_body}"
            )

        raise RuntimeError(f"Callback failed with status {response.status_code}")
