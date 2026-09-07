from fastapi import FastAPI

from smart_home_parser.config import get_settings
from smart_home_parser.schemas import HealthResponse, MetadataResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "REST API for a Transformer-based smart-home command parser. "
        "The initial service exposes health and metadata endpoints."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=HealthResponse, tags=["operations"])
def ready() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse, tags=["operations"])
def metadata() -> MetadataResponse:
    return MetadataResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_version=settings.model_version,
        environment=settings.environment,
        model_loaded=False,
    )
