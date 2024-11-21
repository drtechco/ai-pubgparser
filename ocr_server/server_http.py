from flask import Flask, request, jsonify
import cv2 
import numpy as np
import json



from paddleocr import PaddleOCR

_ocr_en = PaddleOCR(use_angle_cls=False, det=False,
                    lang="en",
                    rec_model_dir="./models/rec_pubg_ppocr_v4_hgnet/",
                    rec_char_dict_path="./pubg_dict.txt",
                    use_gpu=True,
                    rec_thresh=0.75,
                    det_thresh=0.5)
# _ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch", det=True)
app = Flask(__name__)



@app.route("/en", methods=["POST"])
def ocr_en():
    if 'file' not in request.files:
        return jsonify({"what??": "where file"})
    file = request.files["file"].read()
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = _ocr_en.ocr(image, cls=False)
    print("RESULT" , result)
    if len(result) > 0:
        if result[0] is not None:
            text = result[0][0][1][0]
            # add [ if there is a `]` already (common error )
            if "]" in text and not text.startswith("["):
                text = "[" + text
            return jsonify({"string": text})
        else:
            return jsonify({"string": "placeholder"})
    else:
        return jsonify({"string": "placeholder"})



@app.route("/ch", methods=["POST"])
def ocr_ch():
    if 'file' not in request.files:
        return jsonify({"what??": "where file"})
    file = request.files["file"].read()
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = _ocr_en.ocr(image, cls=False)
    if len(result) > 0:
        if result[0] is not None:
            return jsonify({"string": result[0][0][1][0]})
        else:
            return jsonify({"string": "placeholder"})
    else:
        return jsonify({"string": "placeholder"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
