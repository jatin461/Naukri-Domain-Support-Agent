import re
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_version_matches_pin():
    assert (ROOT / ".python-version").read_text().strip().startswith("3.11")
    assert sys.version_info[:2] == (3, 11)


def test_installed_versions_match_requirements_txt():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    assert {"crewai", "autogen-agentchat", "autogen-core", "langchain-core", "chromadb", "sentence-transformers", "fastapi"} <= set(pins)
    for name, pinned in pins.items():
        assert version(name) == pinned, name
    assert version("autogen-agentchat").startswith("0.7."), "an unpinned autogen-agentchat resolves to the legacy 0.2.x line"


def test_lock_file_contains_the_direct_pins():
    lock = (ROOT / "requirements.lock").read_text().lower().replace("_", "-")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            assert f"{m.group(1).lower().replace('_', '-')}=={m.group(2)}" in lock, m.group(1)
