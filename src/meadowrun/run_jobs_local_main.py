import argparse
import datetime
import logging
import os
import subprocess

from meadowrun.run_job_core import CloudProvider


def command_line_main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cloud", choices=CloudProvider)
    parser.add_argument("--cloud-region-name")
    parser.add_argument("--job-id-overrides", required=True)
    parser.add_argument("--job-friendly-name", required=True)
    args = parser.parse_args()

    if bool(args.cloud is None) ^ bool(args.cloud_region_name is None):
        raise ValueError(
            "--cloud and --cloud-region-name must both be provided or both not be "
            "provided"
        )

    job_id_overrides = args.job_id_overrides.split(",")

    environment = os.environ.copy()

    environment["PYTHONUNBUFFERED"] = "1"

    print(f"{datetime.datetime.now()} starting to launch processes")

    for job_id_override in job_id_overrides:
        subprocess.Popen(
            "/var/meadowrun/env/bin/python -m meadowrun.run_job_local_main --job-id "
            f"{args.job_id} --cloud {args.cloud} "
            # cloud parameters are actually optional
            f"--cloud-region-name {args.cloud_region_name} --job-id-override "
            f"{job_id_override} > /var/meadowrun/job_logs/{args.job_friendly_name}.{job_id_override}.log",
            shell=True,
            env=environment
        )

    print(f"{datetime.datetime.now()} done launching processes")


if __name__ == "__main__":
    command_line_main()
