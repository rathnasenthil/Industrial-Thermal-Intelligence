from fastapi import APIRouter

from app.api.routes import alerts, dashboard, events, facilities, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(events.router)
api_router.include_router(facilities.router)
api_router.include_router(alerts.router)
api_router.include_router(dashboard.router)
