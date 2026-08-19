from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    media_root: Path
    config_root: Path
    export_root: Path
    db_path: Path
    scan_interval_seconds: int
    stable_age_seconds: int
    default_before_seconds: float
    default_after_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        config_root = _path("CONFIG_ROOT", "/config")
        return cls(
            media_root=_path("MEDIA_ROOT", "/data"),
            config_root=config_root,
            export_root=_path("EXPORT_ROOT", "/exports"),
            db_path=_path("ANALYZER_DB", str(config_root / "analyzer.db")),
            scan_interval_seconds=max(30, int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))),
            stable_age_seconds=max(0, int(os.getenv("STABLE_AGE_SECONDS", "600"))),
            default_before_seconds=max(0, float(os.getenv("DEFAULT_BEFORE_SECONDS", "8"))),
            default_after_seconds=max(0, float(os.getenv("DEFAULT_AFTER_SECONDS", "5"))),
        )

    def validate(self) -> None:
        if self.media_root == self.export_root:
            raise RuntimeError("MEDIA_ROOT and EXPORT_ROOT must be different")
        if self.media_root in self.export_root.parents:
            raise RuntimeError("EXPORT_ROOT must not be inside the read-only media root")
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)

