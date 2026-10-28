import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app import (
    ActivityDisabledError,
    ActivitySummary,
    Event,
    FileRecord,
    Stats,
    User,
    UserExistsError,
    UserNotFoundError,
)

from .routes import UNAUTHORIZED, ApplicationDep, require_admin
from .schemas import Error, TokenResponse, UserCreate, UserCreated, UserUpdate

DAY = 86400
DEFAULT_WINDOW = 7 * DAY

# Time is Unix seconds everywhere in this API
Since = Annotated[
    float | None,
    Query(description="Start of the period, Unix seconds. Seven days ago when omitted"),
]
Until = Annotated[
    float | None, Query(description="End of the period, Unix seconds. Now when omitted")
]

admin = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    responses={
        **UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN: {
            "model": Error,
            "description": "The token is a user token, not the admin one",
        },
    },
)

USER_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {
        "model": Error,
        "description": "No such user, or it is deleted",
    }
}

ACTIVITY_OFF: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": Error,
        "description": "The activity log is turned off in the config",
    }
}


def user_not_found(user_id: uuid.UUID) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"user not found: {user_id}")


def period(since: float | None, until: float | None) -> tuple[float, float]:
    end = time.time() if until is None else until
    return (end - DEFAULT_WINDOW if since is None else since), end


# --- users


@admin.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    summary="Create a user",
    description="The token in the answer is shown once, only its hash is stored. "
    "Lost token: `POST /admin/users/{id}/token` makes a new one.",
    responses={
        status.HTTP_409_CONFLICT: {
            "model": Error,
            "description": "The name is taken, also by a deleted user",
        },
    },
)
def create_user(body: UserCreate, application: ApplicationDep) -> UserCreated:
    try:
        user, token = application.create_user(body.name, body.allow_embeddings)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from None
    except UserExistsError:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"user already exists: {body.name}"
        ) from None
    return UserCreated(user=user, token=token)


@admin.get(
    "/users", summary="List users", description="Oldest first, deleted users included."
)
def list_users(application: ApplicationDep) -> list[User]:
    return application.list_users()


@admin.patch("/users/{user_id}", summary="Change a user", responses=USER_NOT_FOUND)
def update_user(
    user_id: uuid.UUID, body: UserUpdate, application: ApplicationDep
) -> User:
    try:
        return application.update_user(user_id, body.allow_embeddings)
    except UserNotFoundError:
        raise user_not_found(user_id) from None


@admin.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a user",
    description="Revokes the token at once. The files of the user and the statistics stay.",
    responses=USER_NOT_FOUND,
)
def delete_user(user_id: uuid.UUID, application: ApplicationDep) -> None:
    try:
        application.delete_user(user_id)
    except UserNotFoundError:
        raise user_not_found(user_id) from None


@admin.post(
    "/users/{user_id}/token",
    summary="Make a new token for a user",
    description="The old token stops working at once. For a deleted user this is how to bring it back: "
    "it becomes active again with the new token, `restored` says so. Its files and statistics were never touched.",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": Error, "description": "No such user"}
    },
)
def rotate_token(user_id: uuid.UUID, application: ApplicationDep) -> TokenResponse:
    try:
        token, restored = application.rotate_token(user_id)
        return TokenResponse(token=token, restored=restored)
    except UserNotFoundError:
        raise user_not_found(user_id) from None


# --- statistics


@admin.get(
    "/stats",
    summary="Stored files by user",
    description="How many files and bytes each user uploaded and still has, by image, video and audio. "
    "Users with no files are in the list too. Files with no owner (uploaded by the admin or before users existed) "
    "are listed with a null user id.",
)
def stats(application: ApplicationDep) -> Stats:
    return application.stats()


@admin.get(
    "/activity/summary",
    summary="What happened",
    description="Everything the server saw in the period: requests by action, user, day, transport and IP, "
    "outcomes, timings, traffic, the most popular search queries and the state of background jobs.",
    responses=ACTIVITY_OFF,
)
def activity_summary(
    application: ApplicationDep, since: Since = None, until: Until = None
) -> ActivitySummary:
    start, end = period(since, until)
    try:
        return application.activity_summary(start, end)
    except ActivityDisabledError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "activity log is turned off"
        ) from None


@admin.get(
    "/activity/events",
    summary="Event log",
    description="One entry per request or finished job, newest first.",
    responses=ACTIVITY_OFF,
)
def activity_events(
    application: ApplicationDep,
    since: Since = None,
    until: Until = None,
    user_id: Annotated[uuid.UUID | None, Query(description="Only this user")] = None,
    action: Annotated[
        str | None, Query(description="Handler name, as in the summary")
    ] = None,
    outcome: Annotated[
        str | None, Query(pattern="^(ok|denied|client_error|server_error)$")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Event]:
    start, end = period(since, until)
    try:
        return application.activity_events(
            start, end, user_id, action, outcome, limit, offset
        )
    except ActivityDisabledError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "activity log is turned off"
        ) from None


# --- files


@admin.get(
    "/files",
    summary="Stored files",
    description="Newest first, with the owner, size and upload time.",
)
def list_files(
    application: ApplicationDep,
    owner_id: Annotated[
        uuid.UUID | None, Query(description="Only files of this user")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[FileRecord]:
    return application.list_files(owner_id, limit, offset)


@admin.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a file",
    description="Removes the file from the disk and from the search index.",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": Error, "description": "File not found"}
    },
)
def delete_file(file_id: uuid.UUID, application: ApplicationDep) -> None:
    try:
        application.delete_file(file_id)
    except FileNotFoundError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"file not found: {file_id}"
        ) from None
