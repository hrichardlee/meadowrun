"""
There's no such thing as an "S3 grid job". The code in this module just helps us
implement grid jobs that use an S3-compatible object store to transfer data, e.g. for
EC2 and Kubernetes
"""


from __future__ import annotations

import abc
import asyncio
import hashlib
import io
import json
import os
import pickle
import shutil
import tempfile
import time
from typing import (
    Any,
    AsyncIterable,
    Iterable,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Tuple,
    Type,
)

import aiobotocore.session
import botocore.exceptions
from botocore.exceptions import ClientError

from meadowrun._vendor.fastcdc.fastcdc_py import fastcdc_py
from meadowrun.aws_integration.aws_core import get_bucket_name
from meadowrun.meadowrun_pb2 import ProcessState
from meadowrun.run_job_core import (
    JobCompletion,
    MeadowrunException,
    TaskProcessState,
    WorkerProcessState,
)
from meadowrun.storage_keys import (
    STORAGE_KEY_PROCESS_STATE_SUFFIX,
    STORAGE_KEY_TASK_RESULT_SUFFIX,
    parse_storage_key_process_state,
    parse_storage_key_task_result,
    storage_key_process_state,
    storage_key_task_args,
    storage_key_task_result,
    storage_prefix_outputs,
)

if TYPE_CHECKING:
    import types_aiobotocore_s3
    from typing_extensions import Literal
    from types import TracebackType


class AbstractStorageClient(abc.ABC):
    """
    A wrapper around S3, Azure Storage, Google Cloud Storage, or other similar object
    storage systems.

    All of these systems implement an S3-compatible API, so we could theoretically get
    by with just S3ClientWrapper which uses boto3 under the hood, but the big difference
    is authentication. For example, Google Cloud Storage "natively" uses OAuth2, where
    as S3/boto3 use an HMAC-based authentication scheme. So if we want to use e.g. a
    service account's identity-based permissions on a GKE node, we need to use the
    Google Cloud API directly, as S3ClientWrapper/boto3 will only work with a
    username/password.
    """

    async def __aenter__(self) -> AbstractStorageClient:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    @abc.abstractmethod
    async def get_bytes(self, bucket: str, key: str) -> bytes:
        ...

    @abc.abstractmethod
    async def try_get_bytes(self, bucket: str, key: str) -> Optional[bytes]:
        ...

    async def get_bytes_and_key(self, bucket: str, key: str) -> Tuple[bytes, str]:
        """
        A thin wrapper around get_bytes that also returns the key. Wouldn't be necessary
        if asyncio was easier to work with
        """
        return await self.get_bytes(bucket, key), key

    @abc.abstractmethod
    async def get_byte_range(
        self, bucket: str, key: str, byte_range: Tuple[int, int]
    ) -> bytes:
        ...

    @abc.abstractmethod
    async def write_bytes(self, data: bytes, bucket: str, key: str) -> None:
        ...

    @abc.abstractmethod
    async def write_bytes_if_not_exists(
        self, data: bytes, bucket: str, key: str
    ) -> None:
        ...

    @abc.abstractmethod
    async def get_file(self, bucket: str, key: str, local_filename: str) -> None:
        ...

    @abc.abstractmethod
    async def try_get_file(self, bucket: str, key: str, local_filename: str) -> bool:
        ...

    @abc.abstractmethod
    async def write_file(self, local_filename: str, bucket: str, key: str) -> None:
        ...

    @abc.abstractmethod
    async def list_objects(
        self, bucket: str, key_prefix: str, start_after: str
    ) -> List[str]:
        ...

    @abc.abstractmethod
    async def delete_object(self, bucket: str, key: str) -> None:
        ...


class S3ClientWrapper(AbstractStorageClient):
    """
    This should be used for accessing AWS S3 as well as any other
    username/password-based S3-compatible object storage systems (e.g. Minio)
    """

    def __init__(self, s3_client: types_aiobotocore_s3.S3Client):
        self._s3_client = s3_client

    async def __aenter__(self) -> S3ClientWrapper:
        self._s3_client = await self._s3_client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self._s3_client.__aexit__(exc_type, exc_val, exc_tb)

    async def get_bytes(self, bucket: str, key: str) -> bytes:
        response = await self._s3_client.get_object(Bucket=bucket, Key=key)
        async with response["Body"] as stream:
            return await stream.read()

    async def try_get_bytes(self, bucket: str, key: str) -> Optional[bytes]:
        try:
            return await self.get_bytes(bucket, key)
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            return None

    async def get_byte_range(
        self, bucket: str, key: str, byte_range: Tuple[int, int]
    ) -> bytes:
        response = await self._s3_client.get_object(
            Bucket=bucket, Key=key, Range=f"bytes={byte_range[0]}-{byte_range[1]}"
        )

        async with response["Body"] as stream:
            return await stream.read()

    async def write_bytes(self, data: bytes, bucket: str, key: str) -> None:
        await self._s3_client.put_object(Bucket=bucket, Key=key, Body=data)

    async def _exists(self, bucket: str, key: str) -> bool:
        try:
            await self._s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as error:
            # don't raise an error saying the file doesn't exist
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise error
        return False

    async def write_bytes_if_not_exists(
        self, data: bytes, bucket: str, key: str
    ) -> None:
        if not await self._exists(bucket, key):
            await self.write_bytes(data, bucket, key)

    async def get_file(self, bucket: str, key: str, local_filename: str) -> None:
        response = await self._s3_client.get_object(Bucket=bucket, Key=key)
        async with response["Body"] as stream:
            with open(local_filename, "wb") as f:
                f.write(await stream.read())

    async def try_get_file(self, bucket: str, key: str, local_filename: str) -> bool:
        try:
            await self.get_file(bucket, key, local_filename)
            return True
        except botocore.exceptions.ClientError as error:
            # don't raise an error saying the file doesn't exist
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise error

            return False

    async def write_file(self, local_filename: str, bucket: str, key: str) -> None:
        with open(local_filename, "rb") as f:
            await self._s3_client.put_object(Bucket=bucket, Key=key, Body=f)

    async def list_objects(
        self, bucket: str, key_prefix: str, start_after: str
    ) -> List[str]:
        """Returns keys with the matching prefix in the bucket"""
        paginator = self._s3_client.get_paginator("list_objects_v2")
        results = []
        async for result in paginator.paginate(
            Bucket=bucket, Prefix=key_prefix, StartAfter=start_after
        ):
            for c in result.get("Contents", []):
                results.append(c["Key"])
        return results

    async def delete_object(self, bucket: str, key: str) -> None:
        await self._s3_client.delete_object(Bucket=bucket, Key=key)

    def get_aws_meadowrun_bucket_name(self) -> str:
        """
        This is for S3 clients where we are connecting to the real S3 using the default
        Meadowrun infrastructure for EC2/AWS. In that case, the name of the bucket we
        use can be constructed from the region name. This function should NOT be used if
        we are using an S3 client pointing to a non-S3 object storage.
        """
        return get_bucket_name(self._s3_client.meta.region_name)


def get_storage_client_from_args(
    storage_endpoint_url: Optional[str],
    storage_access_key_id: Optional[str],
    storage_secret_access_key: Optional[str],
) -> S3ClientWrapper:
    kwargs = {}
    if storage_access_key_id is not None:
        kwargs["aws_access_key_id"] = storage_access_key_id
    if storage_secret_access_key is not None:
        kwargs["aws_secret_access_key"] = storage_secret_access_key
    if storage_endpoint_url is not None:
        kwargs["endpoint_url"] = storage_endpoint_url

    # TODO if all the parameters are None then we're implicitly falling back on AWS
    # S3, which we should make explicit
    session = aiobotocore.session.get_session()
    return S3ClientWrapper(session.create_client("s3", **kwargs))  # type: ignore


async def upload_task_args(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    job_id: str,
    args: Iterable[Any],
) -> List[Tuple[int, int]]:
    range_from = 0
    ranges = []
    with io.BytesIO() as buffer:
        for i, arg in enumerate(args):
            pickle.dump(((arg,), {}), buffer)
            range_to = buffer.tell() - 1
            ranges.append((range_from, range_to))
            range_from = range_to + 1

        await storage_client.write_bytes(
            buffer.getvalue(),
            bucket_name,
            storage_key_task_args(job_id),
        )

    return ranges


async def download_task_arg(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    job_id: str,
    byte_range: Tuple[int, int],
) -> Any:
    return await storage_client.get_byte_range(
        bucket_name, storage_key_task_args(job_id), byte_range
    )


async def complete_task(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    job_id: str,
    task_id: int,
    attempt: int,
    process_state: ProcessState,
) -> None:
    """Uploads the result of the task to S3."""
    await storage_client.write_bytes(
        process_state.SerializeToString(),
        bucket_name,
        storage_key_task_result(job_id, task_id, attempt),
    )


async def receive_results(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    job_id: str,
    stop_receiving: asyncio.Event,
    all_workers_exited: asyncio.Event,
    initial_wait_seconds: int = 1,
    receive_message_wait_seconds: int = 20,
    read_worker_process_states: bool = True,
) -> AsyncIterable[Tuple[List[TaskProcessState], List[WorkerProcessState]]]:
    """
    Listens to a result queue until we have results for num_tasks.

    As results become available, yields (task results, worker results). Task results
    will be a list of TaskProcessState. Worker results will be a list of
    WorkerProcessState
    """

    # Behavior is that if stop_receiving is set, we want to return immediately. If
    # all_workers_exited is set, then keep trying for about 3 seconds (just in case some
    # results are still coming in), and then return

    delete_tasks = []

    try:
        results_prefix = storage_prefix_outputs(job_id)
        download_keys_received: Set[str] = set()
        wait = initial_wait_seconds
        workers_exited_wait_count = 0
        while not stop_receiving.is_set() and (
            workers_exited_wait_count < 3 or wait == 0
        ):
            if all_workers_exited.is_set():
                workers_exited_wait_count += 1

            if wait:
                events_to_wait_for = [asyncio.create_task(stop_receiving.wait())]
                if workers_exited_wait_count == 0:
                    events_to_wait_for.append(
                        asyncio.create_task(all_workers_exited.wait())
                    )
                else:
                    # poll more frequently if workers are done, but still wait 1 second
                    # (unless stop_receiving is set)
                    wait = 1

                done, pending = await asyncio.wait(
                    events_to_wait_for,
                    timeout=wait,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                if stop_receiving.is_set():
                    break

            keys = await storage_client.list_objects(bucket_name, results_prefix, "")

            download_tasks = []
            for key in keys:
                if key not in download_keys_received:
                    download_tasks.append(
                        asyncio.create_task(
                            storage_client.get_bytes_and_key(bucket_name, key)
                        )
                    )

            download_keys_received.update(keys)

            if len(download_tasks) == 0:
                if wait == 0:
                    wait = 1
                else:
                    wait = min(wait + 1, receive_message_wait_seconds)
            else:
                wait = 0
                task_results = []
                worker_results = []
                for task_result_future in asyncio.as_completed(download_tasks):
                    process_state_bytes, key = await task_result_future
                    process_state = ProcessState()
                    process_state.ParseFromString(process_state_bytes)
                    if key.endswith(STORAGE_KEY_TASK_RESULT_SUFFIX):
                        task_id, attempt = parse_storage_key_task_result(
                            key, results_prefix
                        )
                        task_results.append(
                            TaskProcessState(task_id, attempt, process_state)
                        )
                    elif key.endswith(STORAGE_KEY_PROCESS_STATE_SUFFIX):
                        if not read_worker_process_states:
                            continue

                        worker_index = parse_storage_key_process_state(
                            key, results_prefix
                        )
                        worker_results.append(
                            WorkerProcessState(worker_index, process_state)
                        )
                    else:
                        print(f"Warning, unrecognized key {key}, will ignore")
                        continue  # don't delete if we can't parse the key

                    delete_tasks.append(
                        asyncio.create_task(
                            storage_client.delete_object(bucket_name, key)
                        )
                    )
                yield task_results, worker_results
    finally:
        await asyncio.gather(*delete_tasks, return_exceptions=True)


async def get_job_completion_from_process_state(
    storage_client: AbstractStorageClient,
    storage_bucket: str,
    job_id: str,
    worker_index: str,
    job_spec_type: Literal["py_command", "py_function", "py_agent"],
    timeout_seconds: int,
    public_address: str,
) -> JobCompletion[Any]:
    """
    Polls for the specified .process_state file, and then creates a JobCompletion object
    based on the process_state

    file_suffix is used by indexed completion jobs to distinguish between the
    completions of different workers, this should be set to the job completion index.
    """

    t0 = time.time()
    process_state_bytes = None
    process_state_key = storage_key_process_state(job_id, worker_index)
    wait = 1
    while time.time() - t0 < timeout_seconds:
        process_state_bytes = await storage_client.try_get_bytes(
            storage_bucket, process_state_key
        )
        if process_state_bytes is not None:
            break
        await asyncio.sleep(wait)
        wait = max(wait + 1, 20)

    if process_state_bytes is None:
        raise TimeoutError(
            f"Waited {timeout_seconds} seconds but {process_state_key} does not exist"
        )

    process_state = ProcessState.FromString(process_state_bytes)
    if process_state.state == ProcessState.ProcessStateEnum.SUCCEEDED:
        # we must have a result from functions, in other cases we can optionally have a
        # result
        if job_spec_type == "py_function" or process_state.pickled_result:
            result = pickle.loads(process_state.pickled_result)
        else:
            result = None
        return JobCompletion(
            result,
            process_state.state,
            process_state.log_file_name,
            process_state.return_code,
            public_address,
        )
    else:
        raise MeadowrunException(process_state)


# This is ugly, but avoids repeated checking if the same bucket+key exists. Only used in
# ensure_uploaded_incremental
_existing_keys: Set[Tuple[str, str]] = set()
_CHUNKS_KEY = "chunks"


async def ensure_uploaded_incremental(
    storage_client: AbstractStorageClient,
    local_file_path: str,
    bucket_name: str,
    key_prefix: str = "",
    avg_chunk_size: int = 1000_000,
) -> str:
    """
    Uploads the specified file from the local machine to S3, in chunks. Each chunk is
    content-hashed and uploaded to "{key_prefix}{hash}.part", where {hash} is a hash of
    the contents of the file. A JSON file with all the parts is uploaded to the key
    "{key_prefix}{hash}.json", where {hash} is a hash of the contents.

    The file is split using content-based-chunking - this means there's no exact size
    for each chunk, but the size can be controlled by setting avg_chunk_size.

    If any S3 object already exists, it is not re-uploaded.

    key_prefix should usually be "" or end in a "/" like "code/".

    Returns the key of the JSON file in the S3 bucket, e.g. "code/123456789abcdefg.json"
    """

    # TODO: only check for existence of chunks if chunks_json does not exist?

    chunk_upload_tasks = []
    keys = []
    for chunk in fastcdc_py(
        local_file_path, avg_size=avg_chunk_size, fat=True, hf=hashlib.blake2b
    ):
        key = f"{key_prefix}{chunk.hash}.part"
        keys.append(key)
        cache_key = (bucket_name, key)
        if (bucket_name, key) not in _existing_keys:
            chunk_upload_tasks.append(
                asyncio.create_task(
                    storage_client.write_bytes_if_not_exists(
                        chunk.data, bucket_name, key
                    )
                )
            )
            _existing_keys.add(cache_key)

    results = await asyncio.gather(*chunk_upload_tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            # on exceptions, we're not sure what the state of the uploaded keys is.
            _existing_keys.clear()
            raise result

    chunks_json = json.dumps({_CHUNKS_KEY: keys}).encode("utf-8")
    chunks_json_key = f"{key_prefix}{hashlib.blake2b(chunks_json).hexdigest()}.json"
    await storage_client.write_bytes_if_not_exists(
        chunks_json, bucket_name, chunks_json_key
    )

    return chunks_json_key


async def download_chunked_file(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    object_name: str,
    file_name: str,
) -> None:
    """Download a file that was uploaded in chunks via ensure_uploaded_incremental"""
    chunk_list_bs = await storage_client.get_bytes(bucket_name, object_name)
    chunks = json.loads(chunk_list_bs.decode("utf-8"))[_CHUNKS_KEY]

    with tempfile.TemporaryDirectory() as tmp:
        await _download_files(storage_client, bucket_name, chunks, tmp)
        _concat_chunks(tmp, chunks, file_name)


def _concat_chunks(
    source_folder: str, chunks: Iterable[str], destination_file: str
) -> None:
    with open(destination_file, "wb") as dest:
        for source_file in chunks:
            with open(
                os.path.join(source_folder, os.path.basename(source_file)), "rb"
            ) as src:
                shutil.copyfileobj(src, dest)


async def _download_files(
    storage_client: AbstractStorageClient,
    bucket_name: str,
    keys: Iterable[str],
    target_folder: str,
) -> None:
    """Download all given keys in the bucket as separate files to the target folder, in
    parallel. Keys can contain duplicates, they'll be downloaded only once.
    """
    download_tasks = []
    for key in set(keys):
        download_tasks.append(
            asyncio.create_task(
                storage_client.get_file(
                    bucket_name, key, os.path.join(target_folder, os.path.basename(key))
                )
            )
        )
    await asyncio.gather(*download_tasks)
