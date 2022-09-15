from __future__ import annotations

import asyncio
import itertools
import json
import os
import pickle
import time
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Callable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import boto3

from meadowrun.aws_integration import s3
from meadowrun.aws_integration.management_lambdas.ec2_alloc_stub import (
    _EC2_ALLOC_TAG,
    _EC2_ALLOC_TAG_VALUE,
)
from meadowrun.meadowrun_pb2 import ProcessState
from meadowrun.shared import pickle_exception

if TYPE_CHECKING:
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_sqs.client import SQSClient

_REQUEST_QUEUE_NAME_PREFIX = "meadowrun-task-"

_T = TypeVar("_T")
_U = TypeVar("_U")


async def _create_request_queue(job_id: str, sqs: SQSClient) -> str:
    """
    See create_queues_and_add_tasks

    Mote that queue names can't contain anything other than letters, numbers, and
    hyphens
    """

    # interestingly, this call will not fail if the exact same queue already exists,
    # but if the queue exists but has different attributes/tags, it will throw an
    # exception
    # TODO try to detect job id collisions?
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sqs.html#SQS.Client.create_queue
    request_queue_response = await sqs.create_queue(
        QueueName=f"{_REQUEST_QUEUE_NAME_PREFIX}{job_id}",
        tags={_EC2_ALLOC_TAG: _EC2_ALLOC_TAG_VALUE},
    )

    # FIFO SQS queues would be nice because they give an "at most once" message delivery
    # guarantee, but we don't need the ordering guarantee, and the performance penalty
    # introduced by the ordering guarantee is unacceptable. It's possible we'll end up
    # executing a task more than once, but according to the docs this should be very
    # rare. If it does happen, then we'll overwrite the result, which is fine.
    # TODO We should put in a check to notify the user that a task ran more than once

    request_queue_url = request_queue_response["QueueUrl"]

    return request_queue_url


def _chunker(it: Iterable[_T], size: int) -> Iterable[Tuple[_T, ...]]:
    """E.g. _chunker([1, 2, 3, 4, 5], 2) -> [1, 2], [3, 4], [5]"""
    iterator = iter(it)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            break
        yield chunk


def _s3_args_key(job_id: str) -> str:
    return f"task-args/{job_id}"


MESSAGE_PREFIX_TASK = "task"
MESSAGE_PREFIX_WORKER_SHUTDOWN = "worker-shutdown"
MESSAGE_GROUP_ID = "_"


async def _add_tasks(
    job_id: str,
    request_queue_url: str,
    s3c: S3Client,
    sqs: SQSClient,
    run_map_args: Iterable[Any],
) -> List[Tuple[int, int]]:
    """
    See create_queues_and_add_tasks

    This can only be called once per request_queue_url. If we wanted to support calling
    add_tasks more than once for the same job, we would need the caller to manage the
    task_ids
    """

    pickles = bytearray()
    grid_tasks: List[str] = []
    range_from = 0
    ranges = []
    for i, arg in enumerate(run_map_args):
        arg_pkl = pickle.dumps(arg)
        pickles.extend(arg_pkl)
        range_to = len(pickles) - 1
        grid_tasks.append(
            json.dumps(
                dict(task_id=i, attempt=1, range_from=range_from, range_end=range_to)
            )
        )
        ranges.append((range_from, range_to))
        range_from = range_to + 1

    await s3.upload_async(
        _s3_args_key(job_id), bytes(pickles), s3c.meta.region_name, s3c
    )

    for tasks_chunk in _chunker(enumerate(grid_tasks), 10):
        # this function can only take 10 messages at a time, so we chunk into
        # batches of 10
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sqs.html#SQS.Client.send_message_batch

        entries = [{"Id": str(i), "MessageBody": task} for i, task in tasks_chunk]

        result = await sqs.send_message_batch(
            QueueUrl=request_queue_url, Entries=entries  # type: ignore
        )
        if "Failed" in result:
            raise ValueError(f"Some grid tasks could not be queued: {result['Failed']}")

    return ranges


async def retry_task(
    request_queue_url: str,
    task_id: int,
    attempt: int,
    range: Tuple[int, int],
    sqs_client: SQSClient,
) -> None:
    """
    See create_queues_and_add_tasks

    This can only be called once per request_queue_url. If we wanted to support calling
    add_tasks more than once for the same job, we would need the caller to manage the
    task_ids
    """

    task = json.dumps(
        dict(
            task_id=task_id,
            attempt=attempt,
            range_from=range[0],
            range_end=range[1],
        )
    )

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sqs.html#SQS.Client.send_message
    await sqs_client.send_message(QueueUrl=request_queue_url, MessageBody=task)


async def _add_worker_shutdown_messages(
    request_queue_url: str,
    num_messages_to_send: int,
    sqs_client: SQSClient,
) -> None:
    messages_sent = 0
    while messages_sent < num_messages_to_send:
        num_messages_to_send_batch = min(10, num_messages_to_send - messages_sent)
        # the id only needs to be unique within a single request
        entries = [
            {"Id": str(i), "MessageBody": "{}"}
            for i in range(num_messages_to_send_batch)
        ]

        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sqs.html#SQS.Client.send_message
        result = await sqs_client.send_message_batch(
            QueueUrl=request_queue_url, Entries=entries  # type: ignore
        )
        if "Failed" in result:
            raise ValueError(
                "Some worker shutdown messages could not be queued: "
                f"{result['Failed']}"
            )

        messages_sent += num_messages_to_send_batch


_GET_TASK_TIMEOUT_SECONDS = 60 * 2  # 2 minutes


def _get_task(
    request_queue_url: str,
    job_id: str,
    region_name: str,
    receive_message_wait_seconds: int,
) -> Optional[Tuple[int, int, bytes]]:
    """
    Gets the next task from the specified request_queue, downloads the task argument
    from S3, and then deletes the message from the request_queue.

    Returns a tuple of task_id and pickled argument if there was a task, otherwise
    returns None. Waits receive_message_wait_seconds for a task.
    """
    sqs = boto3.client("sqs", region_name=region_name)

    # get the task message
    t0 = time.time()
    while True:
        if time.time() - t0 > _GET_TASK_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Waited more than {_GET_TASK_TIMEOUT_SECONDS} for the next task but no"
                " task was available. This is unexpected--the GridJobDriver should have"
                " sent a shutdown message or a SIGINT explicitly"
            )

        result = sqs.receive_message(
            QueueUrl=request_queue_url, WaitTimeSeconds=receive_message_wait_seconds
        )

        if "Messages" in result:
            break

    messages = result["Messages"]
    if len(messages) != 1:
        raise ValueError(f"Requested one message but got {len(messages)}")

    # parse the task request
    task = json.loads(messages[0]["Body"])
    if "task_id" in task:
        task_id, attempt, range_from, range_end = (
            task["task_id"],
            task["attempt"],
            task["range_from"],
            task["range_end"],
        )
        arg = s3.download(_s3_args_key(job_id), region_name, (range_from, range_end))

        # TODO store somewhere (DynamoDB?) that this worker has picked up the task.

        # acknowledge receipt/delete the task request message so we don't have duplicate
        # tasks running
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sqs.html#SQS.Client.delete_message
        sqs.delete_message(
            QueueUrl=request_queue_url, ReceiptHandle=messages[0]["ReceiptHandle"]
        )

        return task_id, attempt, arg

    # it's a worker exit message, so return None to exit
    sqs.delete_message(
        QueueUrl=request_queue_url, ReceiptHandle=messages[0]["ReceiptHandle"]
    )
    return None


def _s3_results_prefix(job_id: str) -> str:
    return f"task-results/{job_id}/"


def _s3_results_key(job_id: str, task_id: int, attempt: int) -> str:
    # A million tasks and 1000 attempts should be enough for everybody. Formatting the
    # task is important because when we task download results from S3, we use the
    # StartFrom argument to S3's ListObjects to exclude most tasks we've already
    # downloaded.
    return f"{_s3_results_prefix(job_id)}{task_id:06d}/{attempt:03d}"


def _s3_result_key_to_task_id_attempt(key: str, results_prefix: str) -> Tuple[int, int]:
    [task_id, attempt] = key.replace(results_prefix, "").split("/")
    return int(task_id), int(attempt)


def _complete_task(
    job_id: str,
    region_name: str,
    task_id: int,
    attempt: int,
    process_state: ProcessState,
) -> None:
    """Uploads the result of the task to S3."""

    process_state_bytes = process_state.SerializeToString()
    s3.upload(
        _s3_results_key(job_id, task_id, attempt), process_state_bytes, region_name
    )


def worker_function(
    function: Callable[[Any], Any],
    request_queue_url: str,
    job_id: str,
    region_name: str,
    public_address: str,
    log_file_name: str,
) -> None:
    """
    Runs a loop that gets tasks off of the request_queue, calls function on the
    arguments of the task, and uploads the results to S3.
    """
    pid = os.getpid()
    log_file_name = f"{public_address}:{log_file_name}"

    while True:
        task = _get_task(
            request_queue_url,
            job_id,
            region_name,
            3,
        )
        if not task:
            break

        task_id, attempt, arg = task
        print(f"Meadowrun agent: About to execute task #{task_id}, attempt #{attempt}")
        try:
            result = function(pickle.loads(arg))
        except Exception as e:
            traceback.print_exc()

            process_state = ProcessState(
                state=ProcessState.ProcessStateEnum.PYTHON_EXCEPTION,
                pid=pid,
                pickled_result=pickle_exception(e, pickle.HIGHEST_PROTOCOL),
                return_code=0,
                log_file_name=log_file_name,
            )
        else:
            process_state = ProcessState(
                state=ProcessState.ProcessStateEnum.SUCCEEDED,
                pid=pid,
                pickled_result=pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
                return_code=0,
                log_file_name=log_file_name,
            )

        print(
            f"Meadowrun agent: Completed task #{task_id}, attempt #{attempt}, "
            f"state {ProcessState.ProcessStateEnum.Name(process_state.state)}"
        )

        _complete_task(
            job_id,
            region_name,
            task_id,
            attempt,
            process_state,
        )


async def receive_results(
    job_id: str,
    region_name: str,
    s3_client: S3Client,
    stop_receiving: asyncio.Event,
    all_workers_exited: asyncio.Event,
    receive_message_wait_seconds: int = 20,
) -> AsyncIterable[List[Tuple[int, int, ProcessState]]]:
    """
    Listens to a result queue until we have results for num_tasks. Returns the unpickled
    results of those tasks.
    """

    # Note: download here is all via S3. It's important that downloads are fast enough -
    # in some cases (many workers, small-ish tasks) the rate of downloading the results
    # to the client can be a limiting factor. Other alternatives that were considered:
    # - SQS queues. Limitations are 1. Only 10 messages can be downloaded at a time,
    #   which slows us down when we have hundreds of tasks completing quickly. 2. SQS
    #   message limit is 256KB which means we need to use S3 to transfer the actual data
    #   in the case of large results.
    # - Sending data via SSH/SCP. This seems like it should be faster especially in the
    #   same VPC/subnet, but even in that case, uploading to S3 first seems to be
    #   faster. It's possible this would make sense in cases where we e.g. overwhelm the
    #   S3 bucket.

    # Behavior is that if stop_receiving is set, we want to return immediately. If
    # all_workers_exited is set, then keep trying for about 3 seconds (just in case some
    # results are still coming in), and then return

    results_prefix = _s3_results_prefix(job_id)
    download_keys_received: Set[str] = set()
    wait = 4  # it's rare for any results to be ready in <4 seconds
    workers_exited_wait_count = 0
    while not stop_receiving.is_set() and (workers_exited_wait_count < 3 or wait == 0):

        if all_workers_exited.is_set():
            workers_exited_wait_count += 1

        if wait:
            events_to_wait_for = [stop_receiving.wait()]
            if workers_exited_wait_count == 0:
                events_to_wait_for.append(all_workers_exited.wait())
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

        keys = await s3.list_objects_async(results_prefix, "", region_name, s3_client)

        download_tasks = []
        for key in keys:
            if key not in download_keys_received:
                download_tasks.append(
                    asyncio.create_task(s3.download_async(key, region_name, s3_client))
                )

        download_keys_received.update(keys)

        if len(download_tasks) == 0:
            if wait == 0:
                wait = 1
            else:
                wait = min(wait * 2, receive_message_wait_seconds)
        else:
            wait = 0
            results = []
            for task_result_future in asyncio.as_completed(download_tasks):
                key, process_state_bytes = await task_result_future
                process_state = ProcessState()
                process_state.ParseFromString(process_state_bytes)
                task_id, attempt = _s3_result_key_to_task_id_attempt(
                    key, results_prefix
                )
                results.append((task_id, attempt, process_state))
            yield results
