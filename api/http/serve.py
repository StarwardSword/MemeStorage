import uvicorn
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from anyio import to_thread
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app import Application, RecordNotFoundError
from .config import Configuration
from .routes import files, search, service

DESCRIPTION = "Meme storage (images, video, audio) with search by text query."


async def not_found(request: Request, exc: Exception) -> JSONResponse:
    ids = exc.target_id if isinstance(exc, RecordNotFoundError) else []
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"file not found: {', '.join(str(x) for x in ids)}"},
    )


def create_app(app: Application, config: Configuration) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        # Handlers are sync and run in this pool, so it bounds concurrent model calls like grpc max_workers
        to_thread.current_default_thread_limiter().total_tokens = config.max_workers
        yield

    api = FastAPI(
        title="MemeStorage",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        exception_handlers={RecordNotFoundError: not_found},
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
