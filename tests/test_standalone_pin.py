"""requirements-standalone.txt installs the llmt-runtime that pyproject.toml's floor asks for (2026.11.18.8: it still
named 2026.11.12.1 after cfdt began importing workers.make_pool, 2026.11.18.5)."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_standalone_runtime_matches_the_floor():
    floor = re.search(r'"llmt-runtime>=([0-9.]+)"', (REPO / "pyproject.toml").read_text()).group(1)
    pin = re.search(r"llmt-runtime@([0-9.]+)", (REPO / "requirements-standalone.txt").read_text()).group(1)
    assert pin == floor
