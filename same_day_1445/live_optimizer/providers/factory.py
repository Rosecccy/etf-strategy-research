from __future__ import annotations

from pathlib import Path

from .auto import AutoMinuteProvider
from .eastmoney import EastmoneyMinuteProvider
from .file_provider import FileMinuteProvider
from .tencent import TencentMinuteProvider


def build_provider(config: dict | None = None):
    config = config or {}
    kind = str(config.get("kind") or "auto").strip().lower()
    if kind == "auto":
        return AutoMinuteProvider()
    if kind == "tencent":
        return TencentMinuteProvider()
    if kind == "eastmoney":
        return EastmoneyMinuteProvider()
    if kind == "file":
        root = str(config.get("file_root") or "").strip()
        if not root:
            raise ValueError("file provider requires file_root")
        return FileMinuteProvider(Path(root))
    raise ValueError(f"unsupported minute provider: {kind}")
