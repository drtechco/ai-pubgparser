from ultralytics import YOLO
import os 
from pathlib import Path
import uuid
import cv2
from tqdm import tqdm


def get_file_by_extension(target_folder: str, extensions: tuple):
    '''
    Takes in a target folder and a tuple of extensions,
    returns a list of path strings of files with the specified extensions,
    including those in subfolders.
    '''
    _a = []
    for root, _, files in os.walk(target_folder):
        for file in files:
            if file.endswith(extensions):
                _a.append(os.path.join(root, file))
    return _a

TARGET_FOLDER = "/home/hbdesk/pubg_parser/stream_saver/outputs/crop2"
DEST_FOLDER = "/home/hbdesk/pubg_parser/stream_saver/outputs/filtered_logs3"
all_images = get_file_by_extension(TARGET_FOLDER, (".png", ".jpg"))
load_yolo = YOLO("/home/hbdesk/pubg_ocr/models/ui_det640.pt")
UI_MODEL_NAMES = load_yolo.names
log_files = []

for img_path in tqdm(all_images):
    img = cv2.imread(img_path)
    _res = load_yolo(img, show=False, verbose=False, conf=0.7)  
    for result in _res:
        classes = result.boxes.cls.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for idx, xyxy in enumerate(xyxys):
            class_name = UI_MODEL_NAMES[classes[idx]]
            x1, y1, x2, y2 = [int(x) for x in xyxy] 
            if class_name == "LOG":
                # _pth = Path(img_path).stem
                _fname = Path(img_path).stem
                _ext = os.path.splitext(img_path)[1]
                _pth = f"{_fname}{_ext}"
                log_files.append(_pth)

for file in tqdm(log_files):
    try:
        tgt = os.path.join(TARGET_FOLDER, file)
        dest = os.path.join(DEST_FOLDER, file)
        os.rename(tgt, dest)
    except Exception as e:
        continue
