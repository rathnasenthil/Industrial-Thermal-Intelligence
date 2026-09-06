from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.firms_nrt_scheduler import (
    start_firms_nrt_scheduler,
    stop_firms_nrt_scheduler,
)

configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Phase 12: start only when FIRMS_NRT_ENABLED=true and not under pytest.
    start_firms_nrt_scheduler(settings)
    try:
        yield
    finally:
        stop_firms_nrt_scheduler()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
