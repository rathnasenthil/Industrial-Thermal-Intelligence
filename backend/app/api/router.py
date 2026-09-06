from fastapi import APIRouter

from app.api.routes import health

api_router = APIRouter()
api_router.include_router(health.router)

# TODO: Register additional routers as features are implemented, e.g.:
#   from app.api.routes import events, alerts
#   api_router.include_router(events.router)
#   api_router.include_router(alerts.router)
