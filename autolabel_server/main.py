from ultralytics import YOLO
import json
import os
import cv2 
import numpy as np
from flask import Flask, request, jsonify, url_for
from annotations import LabelMe
import random
import string

def generate_identifier(length=10):
    if length < 1:
        raise ValueError("Length must be at least 1")
    
    # Define the character set: letters and digits
    characters = string.ascii_letters + string.digits
    
    # Generate the identifier
    identifier = '_' + ''.join(random.choice(characters) for _ in range(length - 1))
    
    return identifier
UI_MODEL_PATH="../webui/models/ui_det640v1.3_L.pt"
LOG_PARSER_PATH="../webui/models/log_parser640v1.2_TEAMS.pt"

ui_model = YOLO(UI_MODEL_PATH)
log_parser = YOLO(LOG_PARSER_PATH)
UI_MODEL_NAMES = ui_model.names
LOG_PARSER = log_parser.names

app = Flask(__name__)

def has_no_empty_params(rule):
    defaults = rule.defaults if rule.defaults is not None else ()
    arguments = rule.arguments if rule.arguments is not None else ()
    return len(defaults) >= len(arguments)


'''
{
  "id": 1294,
  "annotations": [
    {
      "id": 505,
      "completed_by": 1,
      "result": [
        {
          "original_width": 206,
          "original_height": 25,
          "image_rotation": 0,
          "value": {
            "x": 2.862254025044724,
            "y": 9.335718545020862,
            "width": 47.10793082886106,
            "height": 85.98688133571854,
            "rotation": 0,
            "rectanglelabels": [
              "pzone"
            ]
          },
          "id": "_aBxgHwsYJ",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "manual"
        },
        {
          "original_width": 206,
          "original_height": 25,
          "image_rotation": 0,
          "value": {
            "x": 53.07096004770423,
            "y": 19.654144305307096,
            "width": 28.86106141920095,
            "height": 67.80679785330948,
            "rotation": 0,
            "rectanglelabels": [
              "name"
            ]
          },
          "id": "Z2iFGJoKiD",
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "origin": "manual"
        }
      ],
      "was_cancelled": false,
      "ground_truth": false,
      "created_at": "2024-10-29T07:42:34.719700Z",
      "updated_at": "2024-10-29T07:42:34.719732Z",
      "draft_created_at": "2024-10-29T07:42:30.289906Z",
      "lead_time": 15.875,
      "prediction": {},
      "result_count": 0,
      "unique_id": "0a98482a-f74a-4b5c-8ca6-6c4898747240",
      "import_id": null,
      "last_action": null,
      "task": 1294,
      "project": 5,
      "updated_by": 1,
      "parent_prediction": null,
      "parent_annotation": null,
      "last_created_by": null
    }
  ],
  "file_upload": "62be90dd-ed3c73dce8d34f8ab6d93c3b598018d7.png",
  "drafts": [],
  "predictions": [],
  "data": {
    "image": "/data/upload/5/62be90dd-ed3c73dce8d34f8ab6d93c3b598018d7.png"
  },
  "meta": {},
  "created_at": "2024-10-29T07:41:39.638627Z",
  "updated_at": "2024-10-29T07:42:34.817402Z",
  "inner_id": 1,
  "total_annotations": 1,
  "cancelled_annotations": 0,
  "total_predictions": 0,
  "comment_count": 0,
  "unresolved_comment_count": 0,
  "last_comment_updated_at": null,
  "project": 5,
  "updated_by": 1,
  "comment_authors": []
}
'''


'''
{
  "original_width": 206,
  "original_height": 25,
  "image_rotation": 0,
  "value": {
    "x": 2.862254025044724,
    "y": 9.335718545020862,
    "width": 47.10793082886106,
    "height": 85.98688133571854,
    "rotation": 0,
    "rectanglelabels": [
      "pzone"
    ]
  },
  "id": "_aBxgHwsYJ",
  "from_name": "label",
  "to_name": "image",
  "type": "rectanglelabels",
  "origin": "manual"
}
'''

def new_annotation(o_width: int, o_height: int, inf_result: dict):
    return {
                  "original_width": o_width,
                  "original_height": o_height,
                  "image_rotation": 0,
                  "value": inf_result,
                  "id": generate_identifier(),
                  "from_name": "label",
                  "to_name": "image",
                  "type": "rectanglelabels",
                  "origin": "auto_label"
            }

def new_annotation_result(x: float, y: float, w: float, h: float, class_name: str):
    return {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [f"{class_name}"]
        }

@app.route("/", methods=["GET"])
def index():
    return jsonify({"IS_THE_SERVER_UP?":"YES (thats why u see this)",
                    "UI_MODEL":os.path.basename(UI_MODEL_PATH), 
                    "LOG_PARSER_MODEL":os.path.basename(LOG_PARSER_PATH)})



@app.route("/label_ui_lm", methods=["POST"])
def label_ui_LM():
    if 'file' not in request.files:
        return jsonify({"ERROR": "where file"})
    file = request.files["file"].read()
    confidence_threshold = 0.75
    if request.form["confidence_thresh"] is not None:
        confidence_threshold = float(request.form["confidence_thresh"])
    img_filename = request.files["file"].filename 
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = ui_model(image, show=False, verbose=False, conf=confidence_threshold)
    width, height, _ = image.shape
    _annotation = LabelMe(width, height, img_filename)
    print(result[0])
    if len(result) > 0:
        for _res in result:
            classes = _res.boxes.cls.tolist()
            xywh_ = _res.boxes.xyxy.tolist()
            confs = _res.boxes.conf.tolist()
            for idx, xywh in enumerate(xywh_):
                class_name = LOG_PARSER[classes[idx]]
                x1,y1,x2,y2 = [int(x) for x in xywh] 
                _annotation.add_label(class_name, [[x1,y1], [x2,y2]], "rectangle", confs[idx])
    else:
        #return empty
        return jsonify({"results": _annotation.label})
    return jsonify({"results": _annotation.label})
@app.route("/label_log_lm", methods=["POST"])
def label_logs_LM():
    if 'file' not in request.files:
        return jsonify({"ERROR": "where file"})
    file = request.files["file"].read()
    confidence_threshold = 0.75
    if request.form["confidence_thresh"] is not None:
        confidence_threshold = float(request.form["confidence_thresh"])
    img_filename = request.files["file"].filename 
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = log_parser(image, show=False, verbose=False, conf=confidence_threshold)
    width, height, _ = image.shape
    _annotation = LabelMe(width, height, img_filename)
    print(result[0])
    if len(result) > 0:
        for _res in result:
            classes = _res.boxes.cls.tolist()
            xywh_ = _res.boxes.xyxy.tolist()
            confs = _res.boxes.conf.tolist()
            for idx, xywh in enumerate(xywh_):
                class_name = LOG_PARSER[classes[idx]]
                x1,y1,x2,y2 = [int(x) for x in xywh] 
                _annotation.add_label(class_name, [[x1,y1], [x2,y2]], "rectangle", confs[idx])
    else:
        #return empty
        return jsonify({"results": _annotation.label})
    return jsonify({"results": _annotation.label})

@app.route("/label_logs", methods=["POST"])
def label_logs():
    if 'file' not in request.files:
        return jsonify({"ERROR": "where file"})
    file = request.files["file"].read()
    confidence_threshold = 0.75
    if request.form["confidence_thresh"] is not None:
        confidence_threshold = float(request.form["confidence_thresh"])
    if request.form["image_filename"] is None:
        return jsonify({"form error": "ples provide a IMAGE FILE NAME!"})
    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = log_parser(image, show=False, verbose=False, conf=confidence_threshold)
    final_results = []
    if len(result) > 0:
        for _res in result:
            classes = _res.boxes.cls.tolist()
            xywh_ = _res.boxes.xywh.tolist()
            for idx, xywh in enumerate(xywh_):
                class_name = LOG_PARSER[classes[idx]]
                x,y,w,h = [int(x) for x in xywh] 
                resu = new_annotation_result(x,y,w,h, class_name)
                final_results.append(resu)
    if len(final_results) > 0:
        return jsonify({"results":final_results})
    else:
        return jsonify({"results": 0})

@app.route("/label_ui", methods=["POST"])
def label_ui():
    if 'file' not in request.files:
        return jsonify({"ERROR": "where file"})
    file = request.files["file"].read()
    confidence_threshold = 0.75
    if request.form["confidence_thresh"] is not None:
        confidence_threshold = float(request.form["confidence_thresh"])

    np_array = np.frombuffer(file, np.uint8)  # Convert bytes to a numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format
    result = ui_model(image, show=False, verbose=False, conf=confidence_threshold)
    final_results = []
    if len(result) > 0:
        for _res in result:
            classes = _res.boxes.cls.tolist()
            xywh_ = _res.boxes.xywh.tolist()
            for idx, xywh in enumerate(xywh_):
                class_name = UI_MODEL_NAMES[classes[idx]]
                x,y,w,h = [int(x) for x in xywh] 
                resu = new_annotation_result(x,y,w,h, class_name)
                final_results.append(resu)
    if len(final_results) > 0:
        return jsonify({"results":final_results})
    else:
        return jsonify({"results": 0})
               

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=9999)





