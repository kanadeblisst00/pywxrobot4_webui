from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_packages_import_in_clean_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from manager.plugin_base import PluginStateStore; from server import create_app",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
