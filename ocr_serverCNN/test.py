from char_classifier import CharClassifier
import numpy as np
import uuid
from ultralytics import YOLO
import cv2 
import argparse
import time
_a = CharClassifier("./models/char_classifierv1.7_224.pth", 224)
_char_seg = YOLO("./models/pubg_char_detv1.2.pt")

def parse_username(input: str):
    _input = input
    if _input[1] == "[" and _input[0] != "[":
        _input = _input[1:]
    _input = _input.replace("]", "] ")
    return _input                                       


def bgr2_3grey(image: np.ndarray):
    '''
    converts a 3 channel rgb image to one that is grayscale but with 3 channels 
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.merge([gray_image, gray_image, gray_image])
    return gray_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="path to testing image")
    opts = parser.parse_args()
    test_img = cv2.imread(opts.img)
    _char_res = _char_seg(test_img,show=False, verbose=True, conf=0.7, iou=0.6)
    chars = []
    for result in _char_res:
        classes = result.boxes.cls.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for xyxy in xyxys:
            x1,y1,x2,y2 = [int(x) for x in xyxy]
            average_pos = np.sum(xyxy) / 4
            cropped_image = test_img[y1:y2, x1:x2]
            bundle = {average_pos:cropped_image}
            chars.append(bundle)
    sorted_chars = sorted(chars, key=lambda x: list(x.keys())[0])
    final_string = []
    debug_collage = []
    cls_time = time.time()
    for idx, char in enumerate(sorted_chars):
        _key = list(char.keys())[0]
        img = char[_key]
        # _img = bgr2_3grey(img)
        _img = img
        cv2.imwrite(f"./debug_imgs/char_{idx}.png", img)
        t_single = time.time()
        char_class, debug_img = _a.get_char_class(_img, debug=True, confidence_threshold=0.7)
        print("char time: ", (time.time() - t_single) * 1000)
        final_string.append(char_class)
        debug_collage.append(debug_img)
    print("total ocr time: ", (time.time() - cls_time) * 1000)
    debug_collage = np.hstack(debug_collage)
    print(debug_collage.shape)
    cv2.imwrite("debug_imgs/char_collage.png", debug_collage)
    post_process_string = parse_username("".join(final_string))                                     
    print(post_process_string)
