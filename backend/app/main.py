from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "REST API over frozen AIML Stage VI thermal events and Stage I.1 "
        "facilities. risk_score is a decision-support prioritization score, "
        "not a fire probability. Missing STA/environmental evidence is "
        "unavailable, not negative evidence. /api/alerts is an investigation "
        "priority view, not an emergency dispatch system."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
