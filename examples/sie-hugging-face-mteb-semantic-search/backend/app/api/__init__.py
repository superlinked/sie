from fastapi import APIRouter

from .routes import models, generate, chroma, search


api_router = APIRouter()

api_router.include_router(models.router)
api_router.include_router(generate.router)
api_router.include_router(chroma.router)
api_router.include_router(search.router)
