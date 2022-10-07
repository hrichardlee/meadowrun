from __future__ import annotations

import asyncio
import os
import shutil
import sys
from typing import Callable, Awaitable

import filelock

from meadowrun.shared import remove_corrupted_environment
from meadowrun.storage_keys import STORAGE_ENV_CACHE_PREFIX
from meadowrun.deployment.prerequisites import (
    EnvironmentSpecPrerequisites,
    GOOGLE_AUTH_PACKAGE,
)

_POETRY_ENVIRONMENT_TIMEOUT = 10 * 60


async def get_cached_or_create_poetry_environment(
    environment_hash: str,
    project_file_path: str,
    prerequisites: EnvironmentSpecPrerequisites,
    new_environment_path: str,
    try_get_file: Callable[[str, str], Awaitable[bool]],
    upload_file: Callable[[str, str], Awaitable[None]],
) -> str:
    """
    If the desired poetry environment exists, does nothing. If the environment has been
    cached, creates it from the cache. Otherwise creates the environment from scratch
    and caches it. Returns the path to the newly created python interpreter.

    try_get_file and upload_file are for interacting with the cache, see
    compile_environment_spec_locally for more details
    """
    # this assumes that that current version of python is what will be used in
    # create_pip_environment
    new_environment_path = (
        f"{new_environment_path}-{sys.version_info.major}.{sys.version_info.minor}"
    )

    venv_path = os.path.join(new_environment_path, ".venv")
    new_environment_interpreter = os.path.join(venv_path, "bin", "python")

    with filelock.FileLock(f"{new_environment_path}.lock", _POETRY_ENVIRONMENT_TIMEOUT):
        if os.path.exists(new_environment_path):
            return new_environment_interpreter

        remote_cached_file_name = (
            f"{STORAGE_ENV_CACHE_PREFIX}{environment_hash}-{sys.version_info.major}."
            f"{sys.version_info.minor}.tar.gz"
        )
        local_cached_file = f"{new_environment_path}.tar.gz"
        download_succeeded = await try_get_file(
            remote_cached_file_name, local_cached_file
        )
        if download_succeeded:
            try:
                print("Unpacking cached poetry environment")
                os.makedirs(venv_path, exist_ok=True)
                try:
                    # TODO maybe cleaner to use the built-in python tar libraries?
                    return_code = await (
                        await asyncio.create_subprocess_exec(
                            "tar", "-xzf", local_cached_file, "-C", venv_path
                        )
                    ).wait()
                    if return_code != 0:
                        raise ValueError(
                            f"Unpacking cached pip environment {local_cached_file} "
                            f"returned code {return_code}"
                        )
                    return new_environment_interpreter
                except BaseException:
                    remove_corrupted_environment(new_environment_path)
                    raise
            finally:
                try:
                    os.remove(local_cached_file)
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    pass

        print("Creating the poetry environment")
        try:
            await create_poetry_environment(
                project_file_path, new_environment_path, prerequisites
            )
        except BaseException:
            remove_corrupted_environment(new_environment_path)
            raise

        try:
            import venv_pack  # see note on reference in pyproject.toml
        except ImportError:
            print(
                "Warning unable to cache poetry environment because venv_pack is "
                "missing"
            )
            return new_environment_interpreter

        # TODO we shouldn't wait for this to start running the job but we also shouldn't
        # kill the container until this finishes
        print("Caching the poetry environment")
        venv_pack.pack(venv_path, output=local_cached_file)
        await upload_file(local_cached_file, remote_cached_file_name)

        return new_environment_interpreter


# this path depends on the poetry installation script, we have to hope that this never
# changes
_POETRY_PATH = "/root/.local/bin/poetry"


async def create_poetry_environment(
    project_file_path: str,
    new_environment_path: str,
    prerequisites: EnvironmentSpecPrerequisites,
) -> None:
    os.makedirs(new_environment_path, exist_ok=True)

    for file in ("pyproject.toml", "poetry.lock"):
        shutil.copyfile(
            os.path.join(project_file_path, file),
            os.path.join(new_environment_path, file),
        )

    if prerequisites & EnvironmentSpecPrerequisites.GOOGLE_AUTH:
        return_code = await (
            await asyncio.create_subprocess_exec(
                _POETRY_PATH,
                "self",
                "add",
                GOOGLE_AUTH_PACKAGE,
            )
        ).wait()
        if return_code != 0:
            raise ValueError(
                "poetry environment prerequisites installation failed with return code "
                f"{return_code}"
            )

    # this code is roughly equivalent to the code in PoetryDockerfile and
    # PoetryDockerfile
    return_code = await (
        await asyncio.create_subprocess_exec(
            _POETRY_PATH, "install", "--no-root", cwd=new_environment_path
        )
    ).wait()
    if return_code != 0:
        raise ValueError(
            f"poetry environment creation in {new_environment_path} failed with return "
            f"code {return_code}"
        )

    # with --no-root, poetry doesn't install wheel or setuptools
    return_code = await (
        await asyncio.create_subprocess_exec(
            ".venv/bin/python",
            "-m",
            "pip",
            "install",
            "setuptools",
            "wheel",
            cwd=new_environment_path,
        )
    ).wait()
    if return_code != 0:
        raise ValueError(
            f"poetry environment creation in {new_environment_path} failed with return "
            f"code {return_code}"
        )
