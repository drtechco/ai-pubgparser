from char_classifier import CharClassifier
import numpy as np
import uuid
from ultralytics import YOLO
import cv2 
from flask import Flask,request, jsonify
import json

app = Flask(__name__)



_a = CharClassifier("./models/char_classifierv1.4_112.pth",112)
_char_seg = YOLO("./models/pubg_char_detv1.2.pt")

def parse_username(input: str):
    _input = input
    if _input[1] == "[" and _input[0] != "[":
        _input = _input[1:]
    _input = _input.replace("]", "] ")
    return _input                                       


@app.route("/en", methods=["POST"])
def ocr_en():
    if 'file' not in request.files:
        return jsonify({"what??": "where file"})
    file = request.files["file"].read()
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    _char_res = _char_seg(image , show=False, verbose=True, conf=0.7, iou=0.5)
    chars = []
    for result in _char_res:
        classes = result.boxes.cls.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for xyxy in xyxys:
            x1,y1,x2,y2 = [int(x) for x in xyxy]
            average_pos = np.sum(xyxy) / 4
            cropped_image = image[y1:y2, x1:x2]
            bundle = {average_pos:cropped_image}
            chars.append(bundle)
    sorted_chars = sorted(chars, key=lambda x: list(x.keys())[0])
    final_string = []
    debug_collage = []
    for idx, char in enumerate(sorted_chars):
        _key = list(char.keys())[0]
        img = char[_key]
        # cv2.imwrite(f"./debug_imgs/char_{idx}.png", img)
        char_class, debug_img = _a.get_char_class(img, debug=True, confidence_threshold=0.6)
        final_string.append(char_class)
        debug_collage.append(debug_img)
    debug_collage = np.hstack(debug_collage)
    cv2.imwrite("debug_imgs/char_collage.png", debug_collage)
    if len(sorted_chars) > 0:
        post_process_string = parse_username("".join(final_string))                                     
        return jsonify({"string": post_process_string})
    else:
        return jsonify({"string": "placeholder"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
