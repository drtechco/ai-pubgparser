import numpy as np
import uuid
from ultralytics import YOLO
import cv2 
from flask import Flask,request, jsonify
import json



app = Flask(__name__)


_ocr = YOLO("./models/pubg_ocr_v1.4.pt")
OCR_NAMES = _ocr.names

def parse_username(input: str):
    _input = input
    if _input[1] == "[" and _input[0] != "[":
        _input = _input[1:]
    if "]" in _input and not _input.startswith('['):
        _input = "[" + _input 
    _input = _input.replace("]", "] ")
    return _input 

def bgr2_3grey(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.merge([gray_image, gray_image, gray_image])
    return gray_bgr

@app.route("/en", methods=["POST"])
def ocr_en():
    if 'file' not in request.files:
        return jsonify({"what??": "where file"})
    file = request.files["file"].read()
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    image = bgr2_3grey(image)
    _char_res = _ocr(image,show=False, verbose=True, conf=0.5)
    final_string = []
    for result in _char_res:
        classes = result.boxes.cls.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for idx, xyxy in enumerate(xyxys):
            x1,y1,x2,y2 = [int(x) for x in xyxy]
            average_pos = np.sum(xyxy) / 4
            bundle = {average_pos:[OCR_NAMES[classes[idx]]]}
            final_string.append(bundle)
    sorted_chars = sorted(final_string, key=lambda x: list(x.keys())[0])
    values = [list(d.values())[0][0] for d in sorted_chars]
    final = "".join(values)
    final = parse_username(final)
    print("detected: ", final)
    if len(sorted_chars) > 0:
        return jsonify({"string": final})
    else:
        return jsonify({"string": "placeholder"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)

