"""
Writes OpenAPI specification of the HTTP API next to this file

Usage: python -m api.http.openapi
"""

from pathlib import Path
from typing import cast

import yaml

from app import Application

from .config import Configuration
from .serve import create_app

OUTPUT = Path(__file__).with_name("openapi.yaml")


def main() -> None:
    # Schema does not depend on the application instance
    spec = create_app(cast(Application, None), Configuration()).openapi()
    OUTPUT.write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"Written {OUTPUT}")


if __name__ == "__main__":
    main()
