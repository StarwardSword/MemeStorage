import hashlib
import re
import secrets
import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel

TOKEN_PREFIX = "mst_"
USER_NAME = re.compile(r"^[\w.\- ]{1,64}$")


class UserExistsError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


# A token is 256 random bits, so a plain hash is enough: there is nothing to brute force
def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class User(BaseModel):
    id: uuid.UUID
    name: str
    allow_embeddings: bool = False
    created_at: float
    # A deleted user stays in the list: files and statistics keep pointing to it
    deleted_at: float | None = None


class Principal(BaseModel, frozen=True):
    """Who sends the request. The admin has no user id: what it uploads has no owner"""

    user_id: uuid.UUID | None
    name: str
    is_admin: bool = False
    allow_embeddings: bool = False


class UserStore(ABC):
    @abstractmethod
    def add(self, user: User, token_hash: str) -> None:
        """Raises UserExistsError when the name is taken, also by a deleted user"""
        raise NotImplementedError

    @abstractmethod
    def find_by_token(self, token_hash: str) -> User | None:
        """Only users that are not deleted"""
        raise NotImplementedError

    @abstractmethod
    def get(self, id: uuid.UUID) -> User | None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[User]:
        """Oldest first, deleted users included"""
        raise NotImplementedError

    @abstractmethod
    def update(self, id: uuid.UUID, allow_embeddings: bool) -> User:
        """Raises UserNotFoundError for an unknown or deleted user"""
        raise NotImplementedError

    @abstractmethod
    def set_token(self, id: uuid.UUID, token_hash: str) -> bool:
        """A deleted user gets back to life with the new token. True when that happened.
        Raises UserNotFoundError for an unknown id"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: uuid.UUID) -> None:
        """Revokes the token and keeps the row. Raises UserNotFoundError for an unknown or deleted user"""
        raise NotImplementedError
