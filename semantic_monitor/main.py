from __future__ import annotations

import uvicorn

from app.config import AppConfig


if __name__ == "__main__":
    config = AppConfig.from_env()
    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=False)
