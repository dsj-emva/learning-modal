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
    assert "runs" in patterns  # local app data (datasets, runs, registry) never enters the image


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


def test_makefile_docker_recipe_never_echoes_the_password() -> None:
    recipe = re.search(r"^docker:\n((?:\t.*\n?)+)", _read("Makefile"), re.M).group(1)
    for line in recipe.splitlines():
        if "APP_PASSWORD" in line:
            assert line.startswith("\t@"), line
    assert '-e APP_PASSWORD="' not in recipe  # not on the docker command line either


def test_serve_sh_refuses_an_unwritable_data_dir(tmp_path: Path) -> None:
    """Non-root branch of scripts/serve.sh: an unwritable DATA_DIR stops the container with an explanation before
    Streamlit starts. (The root branch chowns /data and drops privileges with setpriv; it needs root and Linux, so it
    is not exercised here.)"""
    import os
    import subprocess

    if os.geteuid() == 0:
        import pytest
        pytest.skip("runs as root: the unwritable-directory branch cannot be reached")
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o555)
    try:
        proc = subprocess.run(["sh", str(ROOT / "scripts" / "serve.sh")], cwd=ROOT, capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], "DATA_DIR": str(data), "PORT": "18999"}, timeout=30)
    finally:
        data.chmod(0o755)
    assert proc.returncode == 1
    assert "is not writable" in proc.stderr and "RAILWAY_RUN_UID=0" in proc.stderr
