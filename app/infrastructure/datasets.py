import tempfile
from pathlib import Path

import httpx


async def download_dataset(download_url: str) -> Path:
    async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
        response = await client.get(download_url, follow_redirects=True)
        response.raise_for_status()

    suffix = ".csv"
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition.lower():
        file_name = (
            content_disposition.split("filename=", maxsplit=1)[1].strip().strip('"')
        )
        resolved_suffix = Path(file_name).suffix.lower()
        if resolved_suffix in {".csv", ".xls", ".xlsx"}:
            suffix = resolved_suffix

    if suffix == ".csv":
        content_type = response.headers.get("content-type", "").lower()
        if "spreadsheetml" in content_type:
            suffix = ".xlsx"
        elif "excel" in content_type:
            suffix = ".xls"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(response.content)
        return Path(temp_file.name)
