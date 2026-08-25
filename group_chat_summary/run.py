from __future__ import annotations

import uvicorn

from app.config import AppSettings


if __name__ == "__main__":
    settings = AppSettings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
