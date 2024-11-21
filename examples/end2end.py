'''
this is an end to end example of 
running all the models on a input image
'''
from pprint import pprint
from common_utils import parse_username, bgr2_3grey, extract_numbers
import argparse 
from ultralytics import YOLO
import numpy as np
import cv2 
import time
import os 
from weapons_classifier import WeaponsClassifier
from char_classifier import CharClassifier
from pixel_filter import PixelClassifier
from kill_log import KillLog

# load models
_ui_model = YOLO("./models/ui_det640v1.5_L.pt")
_log_parser = YOLO("./models/log_parser640v1.3_TEAMS.pt")
_char_det = YOLO("./models/pubg_char_detv1.2.pt")
_weapon_cls = WeaponsClassifier("./models/weapon_classifier.pth")
_char_cls = CharClassifier("./models/char_classifierv1.7_224.pth", 224, "./models/char_list.txt")
_pixel_cls = PixelClassifier("./models/pixel_filter.pth", "./models/pixel_filter_cls.txt") 
UI_MODEL_NAMES = _ui_model.names
LOG_PARSER = _log_parser.names



def char_from_img(input_image: np.ndarray) -> str | None:
    '''
    get string from given image, cropped username log.
    '''
    _char_res = _char_det(input_image, show=False, verbose=True, conf=0.7, iou=0.5)
    chars = []
    for result in _char_res:
        classes = result.boxes.cls.tolist()
        xyxys = result.boxes.xyxy.tolist()
        for xyxy in xyxys:
            x1,y1,x2,y2 = [int(x) for x in xyxy]
            average_pos = np.sum(xyxy) / 4
            cropped_image = input_image[y1:y2, x1:x2]
            bundle = {average_pos:cropped_image}
            chars.append(bundle)
    # sorts the characters from left to right , ordered by position,
    # the reason we dont just use x is that the bounding box is not 
    # "grounded" to the image dimension yet, TODO consider sorting by x by grounding the y axises
    sorted_chars = sorted(chars, key=lambda x: list(x.keys())[0])
    final_string = []
    for idx, char in enumerate(sorted_chars):
        _key = list(char.keys())[0]
        img = char[_key]
        char_class, debug_img = _char_cls.get_char_class(img, debug=True, confidence_threshold=0.6)
        final_string.append(char_class)
    if len(sorted_chars) > 0:
        try:
            post_process_string = parse_username("".join(final_string))                                     
        except Exception as _:
            post_process_string = "".join(final_string)
        return post_process_string
    else:
        return None

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--img", type=str,required=True, help="input image")
    opts = args.parse_args()
    # load image and crop top right,
    final_result = {
            "ALIVE": 0,
            "ASSIST": 0,
            "KILLED": 0,
            "LOGS": []
            } 
    input_image= cv2.imread(opts.img)
    input_image = cv2.resize(input_image, (1920, 1080))
    all_kills = []
    # crop top right
    cropped_frame = input_image[0:640, 1920-640:1920]
    
    # run ui model
    ui_elements = _ui_model(cropped_frame, show=False, verbose=False, conf=0.7, iou=0.6)
    if len(ui_elements) > 0:
        for elements in ui_elements:
            classes = elements.boxes.cls.tolist()
            xyxys = elements.boxes.xyxy.tolist()
            for idx, xyxy in enumerate(xyxys):
                class_name = UI_MODEL_NAMES[classes[idx]]
                x1, y1, x2, y2 = [int(x) for x in xyxy] 
                cropped_image = cropped_frame[y1:y2, x1:x2]

                # if a log is detected , run log_parser
                if class_name == "LOG":
                    _kill_log = KillLog()
                    left_right = [] #left to right usernames , killer -> guy being killed
                    teams_pair = [] # same as top but for team numbers (if availabel)
                    parse_log_res = _log_parser(cropped_image, show=False, verbose=False, conf=0.7, iou=0.6)
                    if len(parse_log_res) > 0:
                        for results in parse_log_res:
                            log_classes = results.boxes.cls.tolist()
                            log_xyxys = results.boxes.xyxy.tolist()
                            for idx , log_xyxy in enumerate(log_xyxys):
                                log_classname = LOG_PARSER[log_classes[idx]]
                                _x1, _y1, _x2, _y2 = [int(x) for x in log_xyxy] 
                                # average position to determine 
                                # the position of the log bbox
                                # relative to the other logs.
                                avg_pos = (_x1+_x2+_y1+_y2) / 4
                                if log_classname == "name":
                                    cropped_log = cropped_image[_y1:_y2, _x1:_x2]
                                    res_string = char_from_img(cropped_log)
                                    if res_string is not None:
                                        left_right.append({"avg_pos": avg_pos, "name": res_string})
                                        left_right = sorted(left_right, key=lambda x: x['avg_pos'])
                                        if len(left_right) > 1:
                                            if left_right[0]["name"] != "placeholder":
                                                _kill_log.set_killer(left_right[0]["name"])
                                            if left_right[0]["name"] != "placeholder":
                                                _kill_log.set_dyer(left_right[1]["name"])
                                            all_kills.append(_kill_log)
                                    else:
                                        continue

                                if log_classname == "team_num":
                                    cropped_log = cropped_image[_y1:_y2, _x1:_x2]
                                    team_no = char_from_img(cropped_log)
                                    print("TEAM_NO: ", team_no)
                                    if team_no is not None:
                                        teams_pair.append({"avg_pos": avg_pos, "num": team_no})
                                        teams_pair = sorted(teams_pair, key=lambda x: x["avg_pos"])
                                        if len(teams_pair) > 1:
                                            _kill_log.set_team_pairs(teams_pair)
                                if log_classname == "weapon":
                                    cropped_weapon = cropped_image[_y1:_y2, _x1:_x2]
                                    weapon = _weapon_cls.get_weapon_class(cropped_weapon)
                                    if weapon == "":
                                        weapon = "N/A"
                                    _kill_log.set_kill_method(weapon)
                                if log_classname == "headshot":
                                    _kill_log.set_headshot(True)

                                if log_classname == "fell":
                                    _kill_log.set_kill_method("fell")

                if class_name == "ALIVE":
                   text =  char_from_img(cropped_image)
                   text = extract_numbers(text)
                   if text is not None:
                       final_result["ALIVE"] = str(text)
                   else:
                       pass
                if class_name == "ASSIST":
                   text =  char_from_img(cropped_image)
                   text = extract_numbers(text)
                   if text is not None:
                       final_result["ASSIST"] = str(text)
                   else:
                       pass
                if class_name == "KILLED":
                   text =  char_from_img(cropped_image)
                   text = extract_numbers(text)
                   if text is not None:
                       final_result["KILLED"] = str(text)
                   else:
                       pass
    final_result["LOGS"] = [x.to_dict() for x in all_kills]
    pprint(final_result)
