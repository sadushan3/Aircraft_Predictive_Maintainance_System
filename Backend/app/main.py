"""FastAPI application entry point for the aircraft monitoring backend."""

from fastapi import FastAPI

from app.routers.Anomaly_Health_Monitering.Routes import router


app = FastAPI(
    title="Aircraft Predictive Maintenance System",
    version="1.0.0",
)

app.include_router(router)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """Return a lightweight API availability response."""
    return {"status": "running"}


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "healthy"}
