from fastapi import FastAPI

from app.api.router import api_router


app = FastAPI(
    title="Dynamic Agentic System",
    version="1.0.0",
)


app.include_router(
    api_router,
    prefix="/api",
)


@app.get("/")
def root():
    return {
        "message": "Dynamic Agentic System API is running"
    }