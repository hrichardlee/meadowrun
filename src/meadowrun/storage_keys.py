from __future__ import annotations

from typing import Tuple


STORAGE_ENV_CACHE_PREFIX = "env_cache/"
STORAGE_CODE_CACHE_PREFIX = "code_cache/"


def storage_key_task_args(job_id: str) -> str:
    return f"inputs/{job_id}.task_args"


def storage_key_ranges(job_id: str) -> str:
    return f"inputs/{job_id}.ranges"


def storage_key_job_to_run(job_id: str) -> str:
    return f"inputs/{job_id}.job_to_run"


def storage_key_function(job_id: str) -> str:
    return f"inputs/{job_id}.function"


def storage_key_function_args(job_id: str) -> str:
    return f"inputs/{job_id}.function_args"


def storage_key_code_zip_file(job_id: str) -> str:
    # this is NOT the actual code, it is a serialized CodeZipFile protobuf that has the
    # specs for the code. This is usually contained in the .job_to_run file, but in some
    # cases that doesn't exist
    return f"inputs/{job_id}.code_zip_file"


def storage_prefix_outputs(job_id: str) -> str:
    return f"outputs/{job_id}/"


def storage_key_task_result(job_id: str, task_id: int, attempt: int) -> str:
    # A million tasks and 1000 attempts should be enough for everybody. Formatting the
    # task is important because when we task download results from S3, we use the
    # StartFrom argument to S3's ListObjects to exclude most tasks we've already
    # downloaded.
    return f"{storage_prefix_outputs(job_id)}{task_id:06d}.{attempt:03d}.taskresult"


def parse_storage_key_task_result(key: str, results_prefix: str) -> Tuple[int, int]:
    """Returns task_id, attempt based on the task result key"""
    [task_id, attempt, _] = key.replace(results_prefix, "").split(".")
    return int(task_id), int(attempt)


def storage_key_process_state(job_id: str, worker_index: str) -> str:
    # this will be changed to
    # {storage_prefix_outputs(job_id)}{worker_index}.process_state in the next commit
    return f"{job_id}{worker_index}.process_state"


def storage_key_state(job_id: str, worker_index: str) -> str:
    return f"{storage_prefix_outputs(job_id)}{worker_index}.state"


def storage_key_result(job_id: str, worker_index: str) -> str:
    return f"{storage_prefix_outputs(job_id)}{worker_index}.result"