from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "semaforos-ia"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/semaforos_ia"

    allowed_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
