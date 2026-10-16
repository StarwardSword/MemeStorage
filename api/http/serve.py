import uvicorn
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from anyio import to_thread
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app import Application, UnsupportedTypeError
from .config import Configuration
from .routes import files, search, service

DESCRIPTION = "Meme storage (images, video, audio) with search by text query."


# Routes reject non-media types before saving, this catches what the model itself refuses
async def unsupported_type(request: Request, exc: Exception) -> JSONResponse:
    content_type = exc.args[0] if exc.args else None
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={"detail": f"unsupported content type {content_type}, expected image/*, video/* or audio/*"},
    )


def create_app(app: Application, config: Configuration) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        # Handlers are sync and run in this pool; model calls are bounded separately by the embedding provider
        to_thread.current_default_thread_limiter().total_tokens = config.max_workers
        yield

    api = FastAPI(
        title="MemeStorage",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        exception_handlers={UnsupportedTypeError: unsupported_type},
        # Operation ids are handler names instead of generated "save_file_files_post"
        generate_unique_id_function=lambda route: route.name,
    )
    api.state.application = app

    for router in (files, search, service):
        api.include_router(router)

    return api


# Blocks until the server is stopped
def serve(app: Application, config: Configuration) -> None:
    server = uvicorn.Server(uvicorn.Config(create_app(app, config), host=config.host, port=config.port))
    server.run()
