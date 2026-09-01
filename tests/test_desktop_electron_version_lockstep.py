import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_desktop_electron_versions_are_exact_and_in_lockstep():
    desktop_package = json.loads(
        (REPO_ROOT / "apps" / "desktop" / "package.json").read_text()
    )
    package_lock = json.loads((REPO_ROOT / "package-lock.json").read_text())

    dependency_version = desktop_package["devDependencies"]["electron"]
    builder_version = desktop_package["build"]["electronVersion"]
    locked_version = package_lock["packages"]["apps/desktop"]["devDependencies"][
        "electron"
    ]

    assert dependency_version == builder_version == locked_version
