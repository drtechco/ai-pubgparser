from entrophy_filter import get_metrics
import gradio as gr
import uuid
import pandas as pd
import os
from game_log import GameLogWatcher  
from ultralytics import YOLO
import cv2
import time
from typing import Generator, Tuple
from kill_log import KillLog
import numpy as np
import requests
import re
from weapons_classifier import WeaponsClassifier

STREAM_LANG =["English", "Chinese"] 

stat = {
        "ALIVE": "0",
        "ASSIST": "0", 
        "KILLED": "0" 
        }

def convert_seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return minutes, remaining_seconds

def extract_numbers(input_string):
    try:
        numbers = re.findall(r'\d+', input_string)
        if not numbers:
            raise ValueError("No numbers found in the input string.")
        return ''.join(numbers)

    except Exception as e:
        # Handle any unexpected errors gracefully
        print(f"Error: {e}")
        return None

def get_ch_string_http(input_image: np.ndarray, stream_language: str):
    success, encoded_image = cv2.imencode('.png', input_image)
    if success:
        files = {'file': encoded_image.tobytes()}
        if stream_language.lower() == "chinese":
            response = requests.post("http://localhost:3000/en", files=files)
        else:
            response = requests.post("http://localhost:3000/en", files=files)
        if response.status_code == 200:
            return response.json()["string"] 
        else: 
            return "placeholder"

def get_string_http(input_image: np.ndarray):
    success, encoded_image = cv2.imencode('.png', input_image)
    if success:
        files = {'file': encoded_image.tobytes()}
        response = requests.post("http://127.0.0.1:3000/en", files=files)
        if response.status_code == 200:
            return response.json()["string"] 
        else: 
            return "placeholder"


def bgr2_3grey(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.merge([gray_image, gray_image, gray_image])
    return gray_bgr

def bgr2_3grey_thresh(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return cv2.merge([thresholded, thresholded, thresholded])
   

def sort_kills(kill_list: list):
    unique_pairs = set()
    unique_data = []
    for entry in kill_list:
        pair = (entry['KILLER'], entry['DEAD'])
        # Check if the pair is unique
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            unique_data.append(entry)
    return unique_data

def process_video_realtime(video_path: str, stream_language: str) -> Generator[Tuple[str, str], None, None]:
    all_kills = []
    print(f"STREAM LANGUAGE: {stream_language}")
    """
    Process video file and yield both real-time and cumulative detections.
    Returns: Tuple of (real-time update, cumulative log)
    """
    _weapon_C = WeaponsClassifier("../weapons_classifier/weapon_classifier.pth")
    ui_model = YOLO("./models/ui_det640v1.5_L.pt")
    log_parser = YOLO("./models/log_parser640v1.3_TEAMS.pt")
    UI_MODEL_NAMES = ui_model.names
    LOG_PARSER = log_parser.names
    kills = []
    deads = []
    k_methods = []
    k_time = []
    kill_images = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 0
    duration_seconds = total_frames / video_fps
    _ret , _frame = cap.read()
    height, width = _frame.shape[:2]
    all_kills = []
    total_frame_count = 0
    # Initialize frame counter and cumulative log
    frame_count = 0
    cumulative_log = []
    start_time = time.time()
    total_frame_count = 0
    # Process each frame
    watcher = GameLogWatcher()
    kill_pair_img = []

    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    all_fps_nums = []

    while cap.isOpened():
        total_frame_count += 1
        frame_count += 1
        fps_frame_count += 1
        success, frame = cap.read()

        # Calculate FPS
        fps_current_time = time.time()
        fps_elapsed_time = fps_current_time - fps_start_time
        if fps_elapsed_time > 1.0:
            current_fps = fps_frame_count / fps_elapsed_time
            all_fps_nums.append(current_fps)
            fps_frame_count = 0
            fps_start_time = fps_current_time

        if not success:
            print("error reading video! / end of video buffer!")
            break 
        if frame_count % 40 == 0:
            frame = cv2.resize(frame, (1920, 1080))
            cropped_frame = frame[0:640, 1920-640:1920]
            # cv2.imwrite("./debug_frame_640.png", cropped_frame)
            new_log, percentage = watcher.detect_new_logs(0.5, cropped_frame)
            if new_log:
                _res = ui_model(cropped_frame, show=False, verbose=False, conf=0.7, iou=0.6)
                if len(_res) > 0: 
                    for result in _res:
                        classes = result.boxes.cls.tolist()
                        xyxys = result.boxes.xyxy.tolist()
                        for idx, xyxy in enumerate(xyxys):
                            class_name = UI_MODEL_NAMES[classes[idx]]
                            x1, y1, x2, y2 = [int(x) for x in xyxy] 
                            cropped_image = cropped_frame[y1:y2, x1:x2]
                            if class_name == "LOG":
                                _kill_log = KillLog()
                                _kill_log.set_framenum(total_frame_count)
                                # cv2.imwrite(f"./debug_logs/{uuid.uuid4().hex}.png", cropped_image)
                                _log_res = log_parser(cropped_image, show=False, verbose=False, conf=0.7)
                                left_right = []
                                metric_pair = []
                                teams_pair = []
                                if len(_log_res) > 0:
                                    for results in _log_res:
                                        log_classes = results.boxes.cls.tolist()
                                        log_xyxys = results.boxes.xyxy.tolist()
                                        for idx , log_xyxy in enumerate(log_xyxys):
                                            log_classname = LOG_PARSER[log_classes[idx]]
                                            print("DETECTED in LOG:", log_classname)
                                            _x1, _y1, _x2, _y2 = [int(x) for x in log_xyxy] 
                                            avg_pos = (_x1+_x2+_y1+_y2) / 4
                                            if log_classname == "name":
                                                cropped_log = cropped_image[_y1:_y2, _x1:_x2]
                                                metrics = get_metrics(cropped_log)
                                                # if metrics["variance"] >= 600 and metrics["laplacian_variance"] >= 600 and metrics["entropy"] > 5.5: 
                                                if True:
                                                    kill_pair_img.append(cropped_log)
                                                    # grey_img = bgr2_3grey(cropped_log)
                                                    cropped_log = cv2.cvtColor(cropped_log, cv2.COLOR_BGR2RGB)
                                                    result_string = get_ch_string_http(cropped_log, stream_language) 
                                                    # result_string = get_ch_string_http(grey_img, stream_language) 
                                                    if result_string is not None:
                                                        left_right.append({"avg_pos": avg_pos , "name": result_string})
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
                                                team_no = get_ch_string_http(cropped_log, stream_language)
                                                print("TEAM_NO: ", team_no)
                                                if team_no is not None:
                                                    teams_pair.append({"avg_pos": avg_pos, "num": team_no})
                                                    teams_pair = sorted(teams_pair, key=lambda x: x["avg_pos"])
                                                    if len(teams_pair) > 1:
                                                        _kill_log.set_team_pairs(teams_pair)
                                            if log_classname == "knock" and _kill_log.kill_method=="":
                                                _kill_log.set_kill_method("knock")
                                            if log_classname == "weapon":
                                                cropped_weapon = cropped_image[_y1:_y2, _x1:_x2]
                                                weapon = _weapon_C.get_weapon_class(cropped_weapon)
                                                if weapon == "":
                                                    weapon = "N/A"
                                                _kill_log.set_kill_method(weapon)
                                            if log_classname == "headshot":
                                                _kill_log.set_headshot(True)
                                            if log_classname == "fell":
                                                _kill_log.set_kill_method("fell")
                                            if log_classname == "pzone":
                                                # add name to playzone hack
                                                _kill_log.set_kill_method("pzone")
                                                _kill_log.set_killer("pzone")
                                                _kill_log.set_dyer("pzone")
                                    if _kill_log.killer != "placeholder" and _kill_log.killer != "": 
                                        if _kill_log.dyer != "placeholder" and _kill_log.dyer != "": 
                                            print(f"KILL DETAILS: {_kill_log.to_dict()}\n", 
                                          end='\r', flush=True)
                                            # all_kills.append(_kill_log)
                                            _kl_dict = _kill_log.to_dict()
                                            total_minutes, total_seconds = convert_seconds_to_min_sec(total_frame_count/video_fps)
                                            elapsed_time = time.time() - start_time
                                            if elapsed_time > 1.0:  
                                                fps = frame_count / elapsed_time
                                                frame_count = 0
                                                start_time = time.time()
                                            detection_str = f"""KILL DETAILS:\nTIME: {total_minutes}:{total_seconds:.2f} \nkiller: {_kl_dict["killer"]}\ndead: {_kl_dict["dyer"]}\nis_headshot: {_kl_dict["headshot"]}\nkill_method: {_kl_dict["kill_method"]}\nframe_num: {total_frame_count}\ncurrent_user_stats:\nALIVE: {stat["ALIVE"]} ASSIST: {stat["ASSIST"]} KILLED: {stat["KILLED"]}\nDEBUG STREAM_LANG: {stream_language}\nFPS: {current_fps}\nAVERAGE_FPS: {np.average(all_fps_nums)}"""
                                                              
                                            all_kills.append(detection_str)
                                            _kl_gradio = _kill_log.to_gradio_log(f"{total_minutes}:{total_seconds:.2f}", metric_pair)
                                            # display_img = np.hstack((kill_pair_img[0], kill_pair_img[1]))
                                            # _kl_gradio["Kill Frame"] = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB) 
                                            cumulative_log.append(_kl_gradio)
                                            cumulative_log = sort_kills(cumulative_log)
                                            display_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                                            yield (
                                                detection_str,  # real-time update
                                                pd.DataFrame(cumulative_log),
                                                kill_pair_img 
                                            )
                            if class_name == "ALIVE":
                               text =  get_ch_string_http(cropped_image, stream_language)
                               text = extract_numbers(text)
                               if text is not None:
                                   stat["ALIVE"] = str(text)
                               else:
                                   pass
                            if class_name == "ASSIST":
                               text =  get_ch_string_http(cropped_image, stream_language)
                               text = extract_numbers(text)
                               if text is not None:
                                   stat["ASSIST"] = str(text)
                               else:
                                   pass
                            if class_name == "KILLED":
                               text =  get_ch_string_http(cropped_image, stream_language)
                               text = extract_numbers(text)
                               if text is not None:
                                   stat["KILLED"] = str(text)
                               else:
                                   pass

    cap.release()
    # Final message
    if frame_count == 0:
        final_message = "No objects detected in video."
        yield (final_message, pd.DataFrame(cumulative_log))
    else:
        completion_message = "\nProcessing done!"
        yield (
            completion_message,
            pd.DataFrame(cumulative_log),
            kill_pair_img
        ) 
    os.remove(video_path)
    save_res = pd.DataFrame(cumulative_log).to_csv("results.csv")
demo = gr.Interface(
    fn=process_video_realtime,
    inputs=[gr.Video(),
            gr.Dropdown(
                choices=STREAM_LANG,
                value="english",
                label="stream language",
                info="select the language for detection type"
                )
            ],
    outputs=[
        gr.Textbox(
            label="Real-time Detections",
            lines=5,
            max_lines=10,
            autoscroll=True
        ),
        gr.Dataframe(headers=["Time","Killer", "Dead", "Kill Method"],
                     interactive=True
                     ),
        # gr.Image(label="Kill Frame",
        #         type="numpy"
        #          )
        gr.Gallery(
            label="Kill Frames",
            show_label=True,
            columns=3,
            rows=None,
            height="auto",
            allow_preview=True,
            preview=True,
            object_fit="contain",
            elem_id="kill-frames-gallery"
            )
    ],
    title="PUBG FPS Logs Parser",
    description="upload a pubg video and this app will parse your kill logs and kill/assist/remaining people count with time stamp",
    examples=[["sample.mp4"]],
    cache_examples=True,
    css="""
        #kill-frames-gallery {
            background-color: #f5f5f5;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #kill-frames-gallery img {
            border-radius: 4px;
            transition: transform 0.2s;
        }
        #kill-frames-gallery img:hover {
            transform: scale(1.02);
        }
    """
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
