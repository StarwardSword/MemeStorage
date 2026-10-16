from typing import Any
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from infrastructure import (
    GemmaConfiguration,
    ChromaConfiguration,
    FileStorageConfiguration,
)
from api import GRPCConfiguration, HTTPConfiguration


class Configuration(BaseSettings):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    model_config = SettingsConfigDict(
        yaml_file="./config.yaml",
    )

    gemma: GemmaConfiguration
    chroma: ChromaConfiguration
    files: FileStorageConfiguration
    grpc: GRPCConfiguration | None = None
    http: HTTPConfiguration | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )
