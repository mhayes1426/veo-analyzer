import os
from pathlib import Path

import pytest


@pytest.fixture
def roots(tmp_path: Path):
    media = tmp_path / "media"
    config = tmp_path / "config"
    exports = tmp_path / "exports"
    for path in (media, config, exports):
        path.mkdir()
    return media, config, exports

