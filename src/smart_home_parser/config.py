from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# what this does is that it loads the settings from the .env file
# and caches it so that it doesn't have to be loaded again and again
# this is useful because loading the settings from the .env file can be slow
# especially if the file is large


class Settings(BaseSettings):
    app_name: str = "Smart Home Command Parser API"
    app_version: str = "0.1.0"
    environment: str = "development"
    model_version: str = "not-loaded"
    model_artifact_dir: str = "artifacts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# this is helpful for setting up Docker and Kubernetes environments
# deployment will later set ENVIRONMENT, MODEL_VERSION, and MODEL_ARTIFACT_DIR
# through environment variables rather than
# hardcoding environment-specific values in Python
