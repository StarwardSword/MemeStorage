from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from anyio import to_thread
from fastapi import FastAPI

from app import Application

from .config import Configuration
from .routes import files, search, service

DESCRIPTION = "Meme storage (images, video, audio) with search by text query."


def create_app(app: Application, config: Configuration) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        # Handlers are sync and run in this pool; model calls are bounded separately by the embedding provider
        to_thread.current_default_thread_limiter().total_tokens = config.max_workers
        yield

    api = FastAPI(
        title="MemeStorage",
        version="2.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        # Operation ids are handler names instead of generated "save_file_files_post"
        generate_unique_id_function=lambda route: route.name,
    )

    # api.add_middleware(CORSMiddleware, allow_origins=["*"])

    api.state.application = app

    for router in (files, search, service):
        api.include_router(router)

    return api


# Blocks until the server is stopped
def serve(app: Application, config: Configuration) -> None:
    server = uvicorn.Server(
        uvicorn.Config(create_app(app, config), host=config.host, port=config.port)
    )
    server.run()
