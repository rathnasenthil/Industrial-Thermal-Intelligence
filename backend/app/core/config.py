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

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
