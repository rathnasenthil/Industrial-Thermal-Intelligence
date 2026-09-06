from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration, populated from environment variables / .env.
    See `.env.example` for the full list of supported variables.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Industrial Fire Intelligence API"
    environment: str = "development"

    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/industrial_fire_db"
    )
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "industrial_fire_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    cors_origins: str = "http://localhost:5173"

    # NASA FIRMS near-real-time Area API (MAP_KEY never hardcoded; load from env).
    # Endpoint shape: /api/area/csv/{MAP_KEY}/{SOURCE}/{AREA_COORDINATES}/{DAY_RANGE}
    # See https://firms.modaps.eosdis.nasa.gov/api/area/
    firms_map_key: str = ""
    firms_base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    # VIIRS NOAA-20 NRT aligns with the project's historical JPSS-1 / N20 VIIRS archive.
    firms_product: str = "VIIRS_NOAA20_NRT"
    # Bounding box as west,south,east,north (FIRMS Area API format). Default ≈ India.
    firms_bbox: str = "68.0,6.0,98.0,37.5"
    # FIRMS Area API allows day_range 1..5 for a single request.
    firms_day_range: int = 2
    firms_timeout_seconds: float = 60.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
