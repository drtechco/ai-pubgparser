# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: ml_service.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'ml_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10ml_service.proto\"\"\n\x0cImageRequest\x12\x12\n\nimage_data\x18\x01 \x01(\x0c\" \n\x0eStringResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"O\n\x0c\x42\x62oxResponse\x12\n\n\x02x1\x18\x01 \x01(\x02\x12\n\n\x02y1\x18\x02 \x01(\x02\x12\n\n\x02x2\x18\x03 \x01(\x02\x12\n\n\x02y2\x18\x04 \x01(\x02\x12\x0f\n\x07obj_cls\x18\x05 \x01(\t2m\n\tMLService\x12\x31\n\x0fStringFromImage\x12\r.ImageRequest\x1a\x0f.StringResponse\x12-\n\rBboxFromImage\x12\r.ImageRequest\x1a\r.BboxResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_IMAGEREQUEST']._serialized_start=20
  _globals['_IMAGEREQUEST']._serialized_end=54
  _globals['_STRINGRESPONSE']._serialized_start=56
  _globals['_STRINGRESPONSE']._serialized_end=88
  _globals['_BBOXRESPONSE']._serialized_start=90
  _globals['_BBOXRESPONSE']._serialized_end=169
  _globals['_MLSERVICE']._serialized_start=171
  _globals['_MLSERVICE']._serialized_end=280
# @@protoc_insertion_point(module_scope)