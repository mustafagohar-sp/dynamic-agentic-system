from fastapi import APIRouter

from app.api.routes import knowledge_bases , versions , chat


api_router = APIRouter()


api_router.include_router(
    knowledge_bases.router,
)

api_router.include_router(
    versions.router,
)

api_router.include_router(
    chat.router,
)