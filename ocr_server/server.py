import grpc 
import numpy as np
import time
from concurrent import futures 
from paddleocr import PaddleOCR
import cv2
# import ..ml_service_pb2_grpc
from ml_service_pb2_grpc import MLServiceServicer, add_MLServiceServicer_to_server
from ml_service_pb2 import StringResponse

ocr = PaddleOCR(use_angle_cls=True, lang='en', det=False)

class MLServiceServicer(MLServiceServicer):
    def StringFromImage(self, request, context):
        image_data = request.image_data
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = ocr.ocr(img, cls=False)
        # print(f"TYPE: {type(result)}")
        if len(result) > 0:
            if result[0] is not None:
                _a = result[0][0][1][0]
                return StringResponse(status=f"{_a}") 
            else:
                return StringResponse(status="name_placeholder") 
        else:
            return StringResponse(status="name_placeholder") 


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port('[::]:9995')
    server.start()    
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
