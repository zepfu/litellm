import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_VERSION = "0.1.15"
COLORAMA_VERSION = "0.4.6"
PIN_PATTERN = re.compile(r"semantic_router==(\d+\.\d+\.\d+)")
COLORAMA_PIN_PATTERN = re.compile(r"colorama==(\d+\.\d+\.\d+)")
VULNERABLE_PIN_PATTERN = re.compile(
    r"semantic_router==0\.1\.(?:[8-9]|1[0-4])\b"
)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_semantic_router_docker_paths_use_fixed_dependency_isolated_pin() -> None:
    install_script = _read("docker/install_auto_router.sh")
    hardened_dockerfile = _read("docker/Dockerfile.non_root")
    normalized_install_script = install_script.replace("\\\n", " ")

    assert PIN_PATTERN.findall(install_script) == [FIXED_VERSION]
    assert PIN_PATTERN.findall(hardened_dockerfile) == [
        FIXED_VERSION,
        FIXED_VERSION,
    ]
    assert COLORAMA_PIN_PATTERN.findall(install_script) == [COLORAMA_VERSION]
    assert COLORAMA_PIN_PATTERN.findall(hardened_dockerfile) == [
        COLORAMA_VERSION,
        COLORAMA_VERSION,
    ]

    assert re.search(
        rf"python -m pip install\s+"
        rf'"semantic_router=={re.escape(FIXED_VERSION)}"\s+'
        rf'"colorama=={re.escape(COLORAMA_VERSION)}"\s+'
        r"--no-deps",
        normalized_install_script,
    )
    assert re.search(
        rf"pip wheel [^\n]*--no-deps "
        rf'"semantic_router=={re.escape(FIXED_VERSION)}" '
        rf'"colorama=={re.escape(COLORAMA_VERSION)}"',
        hardened_dockerfile,
    )
    assert (
        "pip install --no-index --find-links=/wheels/ --no-deps "
        f"semantic_router=={FIXED_VERSION} colorama=={COLORAMA_VERSION}"
        in hardened_dockerfile
    )
    for content in (install_script, hardened_dockerfile):
        assert 'version("semantic-router") == "0.1.15"' in content
        assert 'version("colorama") == "0.4.6"' in content
        assert "import colorama" in content
        assert "from semantic_router.routers import SemanticRouter" in content
        assert "from semantic_router.routers.base import Route" in content

    for relative_path in (
        "docker/install_auto_router.sh",
        "docker/Dockerfile.non_root",
    ):
        assert VULNERABLE_PIN_PATTERN.search(_read(relative_path)) is None


def test_root_and_database_images_use_the_pinned_install_helper() -> None:
    invocation = "./docker/install_auto_router.sh"

    assert invocation in _read("Dockerfile")
    assert invocation in _read("docker/Dockerfile.database")
