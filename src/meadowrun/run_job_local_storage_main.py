"""
This is very similar to run_job_local_main.py but instead of communicating with the
client via the local file system, it communicates via an S3 compatible storage bucket.
"""

import argparse
import asyncio
import logging
import os
import traceback

import filelock

import meadowrun.run_job_local
from meadowrun.func_worker_storage_helper import (
    MEADOWRUN_STORAGE_PASSWORD,
    MEADOWRUN_STORAGE_USERNAME,
)
from meadowrun.s3_grid_job import get_storage_client_from_args, read_storage
from meadowrun.meadowrun_pb2 import ProcessState, Job
from meadowrun.k8s_integration.k8s_main import _JOB_IS_RUNNING_FILE


async def main() -> None:
    with filelock.FileLock(_JOB_IS_RUNNING_FILE, 0):
        logging.basicConfig(level=logging.INFO)

        # parse arguments

        parser = argparse.ArgumentParser()

        # arguments for finding the storage bucket (i.e. S3-compatible store) where we
        # can find inputs and put outputs
        parser.add_argument("--storage-bucket", required=True)
        parser.add_argument("--storage-file-prefix", required=True)
        parser.add_argument("--storage-endpoint-url", required=True)

        args = parser.parse_args()

        storage_bucket: str = args.storage_bucket
        storage_file_prefix: str = args.storage_file_prefix
        suffix = os.environ.get("JOB_COMPLETION_INDEX", "")

        # prepare storage client, filenames and pickle protocol for the result
        storage_username = os.environ.get(MEADOWRUN_STORAGE_USERNAME, None)
        storage_password = os.environ.get(MEADOWRUN_STORAGE_PASSWORD, None)
        async with get_storage_client_from_args(
            args.storage_endpoint_url, storage_username, storage_password
        ) as storage_client:
            meadowrun.func_worker_storage_helper.FUNC_WORKER_STORAGE_CLIENT = (
                storage_client
            )
            meadowrun.func_worker_storage_helper.FUNC_WORKER_STORAGE_BUCKET = (
                storage_bucket
            )

            # run the job
            try:
                job = Job.FromString(
                    await read_storage(
                        storage_client,
                        storage_bucket,
                        f"{storage_file_prefix}.job_to_run",
                    )
                )
                # hypothetically we should make sure MEADOWRUN_STORAGE_USERNAME and
                # MEADOWRUN_STORAGE_PASSWORD should get added to the Job object because
                # if we are running _indexed_map_worker we will need those variables.
                # However, this only happens in the non-container job case, in which
                # case the environment variables should just get inherited by the child
                # process that we start.
                first_state, continuation = await meadowrun.run_job_local.run_local(
                    job, None, False
                )
                # TODO to be analogous to run_job_local_main.py we should write
                # f"{storage_file_prefix}.initial_process_state to the storage bucket
                # here

                if (
                    first_state.state != ProcessState.ProcessStateEnum.RUNNING
                    or continuation is None
                ):
                    await storage_client.put_object(
                        Bucket=storage_bucket,
                        Key=f"{storage_file_prefix}.process_state{suffix}",
                        Body=first_state.SerializeToString(),
                    )
                else:
                    final_process_state = await continuation

                    await storage_client.put_object(
                        Bucket=storage_bucket,
                        Key=f"{storage_file_prefix}.process_state{suffix}",
                        Body=final_process_state.SerializeToString(),
                    )
            except:  # noqa: E722
                # so we know what's wrong
                traceback.print_exc()
                raise


if __name__ == "__main__":
    asyncio.run(main())
