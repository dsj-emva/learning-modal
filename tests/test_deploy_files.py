"""Static checks on the deployment files: Dockerfile, .dockerignore, railway.toml, Makefile targets."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    """Text of a file at the repository root."""
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_base_healthcheck_and_non_root_user() -> None:
    dockerfile = _read("Dockerfile")
    assert re.search(r"^FROM python:3\.11-slim\s*$", dockerfile, re.M)
    assert re.search(r"^HEALTHCHECK .*/_stcore/health", dockerfile, re.M | re.S)
    users = re.findall(r"^USER\s+(\S+)", dockerfile, re.M)
    assert users, "Dockerfile has no USER"
    assert users[-1] not in {"root", "0"}


def test_dockerfile_installs_pinned_requirements_before_copying_source() -> None:
    dockerfile = _read("Dockerfile")
    install = dockerfile.index("pip install --no-cache-dir -r requirements.txt")
    assert install < dockerfile.index("COPY emva/")


def test_dockerignore_excludes_ground_truth_and_env() -> None:
    patterns = {line.strip() for line in _read(".dockerignore").splitlines() if line.strip()}
    assert "data/**/ground_truth*" in patterns
    assert ".env" in patterns


def test_railway_toml_parses_with_streamlit_healthcheck() -> None:
    config = tomllib.loads(_read("railway.toml"))
    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["deploy"]["healthcheckPath"] == "/_stcore/health"
    assert config["deploy"]["restartPolicyType"] == "ON_FAILURE"
    assert config["deploy"]["startCommand"] in _read("Dockerfile").replace('", "', " ")


def test_makefile_has_app_and_docker_targets() -> None:
    makefile = _read("Makefile")
    assert re.search(r"^app:", makefile, re.M)
    assert re.search(r"^docker:", makefile, re.M)
    phony = re.search(r"^\.PHONY:(.*)$", makefile, re.M)
    assert phony and {"app", "docker"} <= set(phony.group(1).split())
