# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import ml_service_pb2 as ml__service__pb2

GRPC_GENERATED_VERSION = '1.67.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in ml_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class MLServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.StringFromImage = channel.unary_unary(
                '/MLService/StringFromImage',
                request_serializer=ml__service__pb2.ImageRequest.SerializeToString,
                response_deserializer=ml__service__pb2.StringResponse.FromString,
                _registered_method=True)
        self.BboxFromImage = channel.unary_unary(
                '/MLService/BboxFromImage',
                request_serializer=ml__service__pb2.ImageRequest.SerializeToString,
                response_deserializer=ml__service__pb2.BboxResponse.FromString,
                _registered_method=True)


class MLServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def StringFromImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BboxFromImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MLServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'StringFromImage': grpc.unary_unary_rpc_method_handler(
                    servicer.StringFromImage,
                    request_deserializer=ml__service__pb2.ImageRequest.FromString,
                    response_serializer=ml__service__pb2.StringResponse.SerializeToString,
            ),
            'BboxFromImage': grpc.unary_unary_rpc_method_handler(
                    servicer.BboxFromImage,
                    request_deserializer=ml__service__pb2.ImageRequest.FromString,
                    response_serializer=ml__service__pb2.BboxResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'MLService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('MLService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class MLService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def StringFromImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MLService/StringFromImage',
            ml__service__pb2.ImageRequest.SerializeToString,
            ml__service__pb2.StringResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def BboxFromImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MLService/BboxFromImage',
            ml__service__pb2.ImageRequest.SerializeToString,
            ml__service__pb2.BboxResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
