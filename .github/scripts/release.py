#!/usr/bin/env python3
"""Create the next patch release on GitHub.

Tags follow semantic versioning without a leading "v" (e.g. 0.1.1). Dev
pre-releases (".devN") are ignored when computing the next patch.
"""

from subprocess import PIPE, CalledProcessError, run
from typing import List

INITIAL_VERSION = "0.1.1"


def _run(cmd: List[str]):
    return run(cmd, check=False, stdout=PIPE, stderr=PIPE, text=True)


def get_last_version() -> str:
    """Return the most recent non-dev tag, if any."""
    tag_result = _run(["git", "tag", "--list", "*.*.*", "--sort=-v:refname"])
    if tag_result.returncode == 0:
        for tag in tag_result.stdout.splitlines():
            tag = tag.strip()
            if tag and ".dev" not in tag:
                return tag

    release_result = _run(["gh", "release", "list", "--limit", "20"])
    if release_result.returncode == 0:
        for line in release_result.stdout.splitlines():
            fields = line.split("\t")
            tag = fields[2] if len(fields) > 2 else ""
            if tag and ".dev" not in tag:
                return tag

    return ""


def bump_patch_number(version_number: str) -> str:
    major, minor, patch = version_number.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def create_new_patch_release():
    last_version_number = get_last_version()
    new_version_number = (
        bump_patch_number(last_version_number) if last_version_number else INITIAL_VERSION
    )

    print(f"Creating release {new_version_number}")
    try:
        run(
            ["gh", "release", "create", new_version_number, "--generate-notes", "--latest"],
            check=True,
        )
    except CalledProcessError as err:
        print(err.stderr)
        raise


if __name__ == "__main__":
    create_new_patch_release()
