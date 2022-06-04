"""
Credentials can be managed in a few different ways. Let's say we have a
username/password that we need to access a Docker container registry. The simplest thing
to do is for the user to send the username/password to the coordinator, which then sends
it to agents when the coordinator assigns them a job that needs it. Alternatively, we
could just send over the secret if the agent actually needs it, i.e. the image is not in
the agent machine's local cache.

It's probably better for the user to send over a credentials source, e.g. an AWS secret
or even just a file path that is restricted in some way. If the user is accessing the
coordinator over the public internet, this makes it possible to avoid sending the
username/password over the public internet (even if it's encrypted). Also, this makes it
possible to rotate secrets without manually updating the coordinator. If we take this
strategy, then we need to decide whether the coordinator should send over the actual
credentials or the credentials source to the agents. Sending the actual credentials
exposes them over the wire (even if they're encrypted), but sending the credentials
source means that all of the agents need direct access to the secrets. If e.g. an AWS
IAM role is used to restrict access to an AWS secret, then this is not good because the
job then gets full access to that AWS secret.

TODO actually encrypt the credentials as we send them over?
"""
from __future__ import annotations

import abc
import dataclasses
import json
from typing import Union, Optional, Dict, List, Tuple

import boto3
from typing_extensions import Literal

import meadowrun.docker_controller
from meadowrun.aws_integration.aws_core import _get_default_region_name
from meadowrun.azure_integration.mgmt_functions.azure.azure_rest_api import (
    azure_rest_api,
)
from meadowrun.meadowrun_pb2 import (
    AwsSecret,
    AzureSecret,
    Credentials,
    ServerAvailableFile,
)

# Represents a way to get credentials
CredentialsSource = Union[AwsSecret, AzureSecret, ServerAvailableFile]
# Represents a service that credentials can be used for
CredentialsService = Literal["DOCKER", "GIT"]


@dataclasses.dataclass(frozen=True)
class CredentialsSourceForService:
    """A CredentialsSource with metadata about what service it should be used for"""

    service: CredentialsService
    service_url: str
    source: CredentialsSource


# Maps from a Credentials.Service to a list of (service_url, CredentialsSource). This is
# usually how we'll store a set of CredentialsSources
CredentialsDict = Dict[
    "Credentials.Service.ValueType", List[Tuple[str, CredentialsSource]]
]


class RawCredentials(abc.ABC):
    """
    Represents credentials that can be used immediately (i.e. as opposed to a
    CredentialsSource)
    """

    pass


@dataclasses.dataclass()
class UsernamePassword(RawCredentials):
    username: str
    password: str


@dataclasses.dataclass()
class SshKey(RawCredentials):
    """Should be the content of the private key file generated by e.g. ssh-keygen"""

    private_key: str


async def _get_credentials_from_source(source: CredentialsSource) -> RawCredentials:
    if isinstance(source, AwsSecret):
        # TODO not sure if it's better to try to reuse a client/session or just create a
        #  new one each time? This seems related:
        #  https://github.com/boto/botocore/issues/619
        secret = json.loads(
            boto3.client(
                service_name="secretsmanager",
                region_name=await _get_default_region_name(),
            ).get_secret_value(SecretId=source.secret_name)["SecretString"]
        )
        if source.credentials_type == Credentials.Type.USERNAME_PASSWORD:
            return UsernamePassword(secret["username"], secret["password"])
        elif source.credentials_type == Credentials.Type.SSH_KEY:
            return SshKey(secret["private_key"])
        else:
            raise ValueError(f"Unknown credentials type {source.credentials_type}")
    elif isinstance(source, AzureSecret):
        result = await azure_rest_api(
            "GET",
            f"secrets/{source.secret_name}",
            "7.3",
            base_url=source.vault_name,
            token_scope="https://vault.azure.net/.default",
        )
        secret_value = result["value"]
        if source.credentials_type == Credentials.Type.USERNAME_PASSWORD:
            secret_json = json.loads(secret_value)
            return UsernamePassword(secret_json["username"], secret_json["password"])
        elif source.credentials_type == Credentials.Type.SSH_KEY:
            return SshKey(secret_value)
        else:
            raise ValueError(f"Unknown credentials type {source.credentials_type}")
    elif isinstance(source, ServerAvailableFile):
        if source.credentials_type == Credentials.Type.USERNAME_PASSWORD:
            with open(source.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) != 2:
                raise ValueError(
                    "ServerAvailableFile for credentials must have exactly two lines, "
                    "one for username, and one for password"
                )
            # strip just the trailing newlines, other whitespace might be needed
            lines = [line.rstrip("\r\n") for line in lines]

            return UsernamePassword(lines[0], lines[1])
        elif source.credentials_type == Credentials.Type.SSH_KEY:
            with open(source.path, "r", encoding="utf-8") as f:
                return SshKey(f.read())
        else:
            raise ValueError(f"Unknown credentials type {source.credentials_type}")
    else:
        raise ValueError(f"Unknown type of credentials source {type(source)}")


async def get_matching_credentials(
    service: Credentials.Service.ValueType,
    service_url_to_match: str,
    credentials_dict: CredentialsDict,
) -> Optional[RawCredentials]:
    """
    Gets the credentials where the service_url matches our normalized_url, and then pick
    the longest one (if it exists). max picks the first element it sees, and credentials
    are in the order they were added, so we reverse the list of credentials first so
    that we get the latest set of credentials if multiple credentials are the longest
    """
    # TODO we should clean up "replaced" credentials that will never get used again
    source = max(
        (
            (service_url, source)
            for service_url, source in reversed(credentials_dict.get(service, ()))
            if service_url_to_match.startswith(service_url)
        ),
        key=lambda c: len(c[0]),
        default=None,
    )
    if source is not None:
        return await _get_credentials_from_source(source[1])
    else:
        return None


async def get_docker_credentials(
    repository: str, credentials_dict: CredentialsDict
) -> Optional[RawCredentials]:
    """Get credentials for a docker repository"""

    registry_domain, repository_name = meadowrun.docker_controller.get_registry_domain(
        repository
    )
    return await get_matching_credentials(
        Credentials.Service.DOCKER,
        f"{registry_domain}/{repository_name}",
        credentials_dict,
    )
