from pydantic import BaseModel


# response returned by the health-check endpoint.
class HealthResponse(BaseModel):
    status: str


# response containing application and model runtime information.
class MetadataResponse(BaseModel):
    app_name: str
    app_version: str
    model_version: str
    environment: str
    model_loaded: bool


# ParseCommandRequest
# ParsedCommand
# DeviceExecutionResult
# ParseCommandResponse
