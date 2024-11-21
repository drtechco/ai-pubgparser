from ultralytics import YOLO
import uuid
import cv2
import time



ui_model = YOLO("../webui/models/ui_det640.pt")
log_parser = YOLO("../webui/models/log_parser640.pt")

cap = cv2.VideoCapture("/home/hbdesk/pubg_ocr/vid10/ch_vid10.mp4")

# cap = cv2.VideoCapture("/home/hbdesk/Videos/test_pubg2.mp4")
UI_MODEL_NAMES = ui_model.names
LOG_PARSER = log_parser.names


fps = 0
frame_count = 0
start_time = time.time()

_ret , _frame = cap.read()
height, width = _frame.shape[:2]
all_kills = []
total_frame_count = 0

while cap.isOpened():
    frame_count += 1
    total_frame_count += 1
    ret, frame = cap.read()
    if not ret:
        print("cannot get frame!")
        break
    cropped_frame = frame[0:640, width-640:width]
    if frame_count % 20 == 0:
        _res = ui_model(cropped_frame, show=False, verbose=False, conf=0.8)
        if len(_res) > 0:
            for result in _res:
                classes = result.boxes.cls.tolist()
                xyxys = result.boxes.xyxy.tolist()
                for idx, xyxy in enumerate(xyxys):
                    class_name = UI_MODEL_NAMES[classes[idx]]
                    x1, y1, x2, y2 = [int(x) for x in xyxy] 
                    cropped_image = cropped_frame[y1:y2, x1:x2]
                    if class_name == "LOG":
                        _log_res = log_parser(cropped_image, show=False, verbose=False, conf=0.8)
                        left_right = []
                        if len(_log_res) > 0:
                            for results in _log_res:
                                log_classes = results.boxes.cls.tolist()
                                log_xyxys = results.boxes.xyxy.tolist()
                                for idx, log_xyxy in enumerate(log_xyxys):
                                    log_classname = LOG_PARSER[log_classes[idx]]
                                    _x1, _y1, _x2, _y2 = [int(x) for x in log_xyxy] 
                                    avg_pos = (_x1+x2+y1+y2) / 4
                                    if log_classname == "name":
                                        print("NAME")
                                        cropped_log = cropped_image[_y1:_y2, _x1:_x2]
                                        cv2.imwrite(f"names/name_{uuid.uuid4().hex}.png", cropped_log)
                                    else:
                                        continue
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    cv2.imshow("cropped video", cropped_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
