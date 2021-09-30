# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from nextbeat.server import nextbeat_pb2 as nextbeat_dot_server_dot_nextbeat__pb2


class NextBeatServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.add_jobs = channel.unary_unary(
            "/nextbeat.NextBeatServer/add_jobs",
            request_serializer=nextbeat_dot_server_dot_nextbeat__pb2.AddJobsRequest.SerializeToString,
            response_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.AddJobsResponse.FromString,
        )
        self.instantiate_scopes = channel.unary_unary(
            "/nextbeat.NextBeatServer/instantiate_scopes",
            request_serializer=nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesRequest.SerializeToString,
            response_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesResponse.FromString,
        )
        self.get_events = channel.unary_unary(
            "/nextbeat.NextBeatServer/get_events",
            request_serializer=nextbeat_dot_server_dot_nextbeat__pb2.EventsRequest.SerializeToString,
            response_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.Events.FromString,
        )
        self.register_job_runner = channel.unary_unary(
            "/nextbeat.NextBeatServer/register_job_runner",
            request_serializer=nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerRequest.SerializeToString,
            response_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerResponse.FromString,
        )
        self.manual_run = channel.unary_unary(
            "/nextbeat.NextBeatServer/manual_run",
            request_serializer=nextbeat_dot_server_dot_nextbeat__pb2.ManualRunRequest.SerializeToString,
            response_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.ManualRunResponse.FromString,
        )


class NextBeatServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def add_jobs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def instantiate_scopes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_events(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def register_job_runner(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def manual_run(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_NextBeatServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "add_jobs": grpc.unary_unary_rpc_method_handler(
            servicer.add_jobs,
            request_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.AddJobsRequest.FromString,
            response_serializer=nextbeat_dot_server_dot_nextbeat__pb2.AddJobsResponse.SerializeToString,
        ),
        "instantiate_scopes": grpc.unary_unary_rpc_method_handler(
            servicer.instantiate_scopes,
            request_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesRequest.FromString,
            response_serializer=nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesResponse.SerializeToString,
        ),
        "get_events": grpc.unary_unary_rpc_method_handler(
            servicer.get_events,
            request_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.EventsRequest.FromString,
            response_serializer=nextbeat_dot_server_dot_nextbeat__pb2.Events.SerializeToString,
        ),
        "register_job_runner": grpc.unary_unary_rpc_method_handler(
            servicer.register_job_runner,
            request_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerRequest.FromString,
            response_serializer=nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerResponse.SerializeToString,
        ),
        "manual_run": grpc.unary_unary_rpc_method_handler(
            servicer.manual_run,
            request_deserializer=nextbeat_dot_server_dot_nextbeat__pb2.ManualRunRequest.FromString,
            response_serializer=nextbeat_dot_server_dot_nextbeat__pb2.ManualRunResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "nextbeat.NextBeatServer", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class NextBeatServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def add_jobs(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/nextbeat.NextBeatServer/add_jobs",
            nextbeat_dot_server_dot_nextbeat__pb2.AddJobsRequest.SerializeToString,
            nextbeat_dot_server_dot_nextbeat__pb2.AddJobsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def instantiate_scopes(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/nextbeat.NextBeatServer/instantiate_scopes",
            nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesRequest.SerializeToString,
            nextbeat_dot_server_dot_nextbeat__pb2.InstantiateScopesResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_events(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/nextbeat.NextBeatServer/get_events",
            nextbeat_dot_server_dot_nextbeat__pb2.EventsRequest.SerializeToString,
            nextbeat_dot_server_dot_nextbeat__pb2.Events.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def register_job_runner(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/nextbeat.NextBeatServer/register_job_runner",
            nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerRequest.SerializeToString,
            nextbeat_dot_server_dot_nextbeat__pb2.RegisterJobRunnerResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def manual_run(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/nextbeat.NextBeatServer/manual_run",
            nextbeat_dot_server_dot_nextbeat__pb2.ManualRunRequest.SerializeToString,
            nextbeat_dot_server_dot_nextbeat__pb2.ManualRunResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
