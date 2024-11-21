import grpc 
from ml_service_pb2_grpc import MLServiceServicer, MLServiceStub, add_MLServiceServicer_to_server
from ml_service_pb2 import ImageRequest

def run():
    with open("/home/hbdesk/pubg_ocr/debug_output/LOG_0.png", "rb") as _d:
        image_data = _d.read()
    channel = grpc.insecure_channel("localhost:9995")
    stub = MLServiceStub(channel)
    request = ImageRequest(image_data=image_data)
    print(f"R: {type(request)}")
    response = stub.StringFromImage(request)
    print(response)
run()
