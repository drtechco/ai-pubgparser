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
            response = requests.post("http://127.0.0.1:3000/ch", files=files)
        else:
            response = requests.post("http://127.0.0.1:3000/en", files=files)
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



def process_video_realtime(video_path: str, stream_language: str) -> Generator[Tuple[str, str], None, None]:
    all_kills = []
    print(f"STREAM LANGUAGE: {stream_language}")
    """
    Process video file and yield both real-time and cumulative detections.
    Returns: Tuple of (real-time update, cumulative log)
    """
    ui_model = YOLO("./models/ui_det640.pt")
    log_parser = YOLO("./models/log_parser640.pt")
    UI_MODEL_NAMES = ui_model.names
    LOG_PARSER = log_parser.names
    kills = []
    deads = []
    k_methods = []
    k_time = []

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

    while cap.isOpened():
        total_frame_count += 1
        frame_count += 1
        success, frame = cap.read()
        if not success:
            print("error reading video!")
            break
        cropped_frame = frame[0:640, width-640:width]
        if frame_count % 60 == 0:
            new_log, percentage = watcher.detect_new_logs(0.5, cropped_frame)
            # new_log = True
            if new_log:
                _res = ui_model(cropped_frame, show=False, verbose=False, conf=0.73)
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
                                _log_res = log_parser(cropped_image, show=False, verbose=False, conf=0.7)
                                left_right = []
                                if len(_log_res) > 0:
                                    for results in _log_res:
                                        log_classes = results.boxes.cls.tolist()
                                        log_xyxys = results.boxes.xyxy.tolist()
                                        for idx , log_xyxy in enumerate(log_xyxys):
                                            log_classname = LOG_PARSER[log_classes[idx]]
                                            _x1, _y1, _x2, _y2 = [int(x) for x in log_xyxy] 
                                            avg_pos = (_x1+_x2+_y1+_y2) / 4
                                            if log_classname == "name":
                                                cropped_log = cropped_image[_y1:_y2, _x1:_x2]
                                                cv2.imwrite(f"./debug_logs/{uuid.uuid4().hex}.png", cropped_log) 
                                                result_string = get_ch_string_http(cropped_log, stream_language) 
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
                                            if log_classname == "weapon":
                                                _kill_log.set_kill_method("weapon")
                                            if log_classname == "headshot":
                                                _kill_log.set_headshot(True)
                                            if log_classname == "fell":
                                                _kill_log.set_kill_method("fell")
                                            if log_classname == "pzone":
                                                # add name to playzone hack
                                                _kill_log.set_kill_method("pzone")
                                                _kill_log.set_killer("pzone")
                                                _kill_log.set_dyer("pzone")
                                            if log_classname == "knock":
                                                _kill_log.set_kill_method("knock")
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
                                            detection_str = f"""KILL DETAILS:\nTIME: {total_minutes}:{total_seconds:.2f} \nkiller: {_kl_dict["killer"]}\ndead: {_kl_dict["dyer"]}\nis_headshot: {_kl_dict["headshot"]}\nkill_method: {_kl_dict["kill_method"]}\nframe_num: {total_frame_count}\ncurrent_user_stats:\nALIVE: {stat["ALIVE"]} ASSIST: {stat["ASSIST"]} KILLED: {stat["KILLED"]}\nDEBUG STREAM_LANG: {stream_language}\n"""
                                                              
                                            all_kills.append(detection_str)
                                            _kl_gradio = _kill_log.to_gradio_log(f"{total_minutes}:{total_seconds:.2f}")
                                            cumulative_log.append(_kl_gradio)
                                            yield (
                                                detection_str,  # real-time update
                                                pd.DataFrame(cumulative_log)  # complete log
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
        # cumulative_log.append(completion_message)
        yield (
            completion_message,
            pd.DataFrame(cumulative_log)
        ) 
    os.remove(video_path)
    # save_res = pd.DataFrame(cumulative_log).to_csv("results.csv")
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
        # gr.Textbox(
        #     label="Complete Log",
        #     lines=10,
        #     max_lines=30,
        #     autoscroll=True
        # )
        gr.Dataframe(headers=["Time", "Killer", "Dead", "Kill Method"], interactive=True),
        gr.Image(
            label="Kill Frame",
            type="numpy"
        )
    ],
    title="PUBG FPS Logs Parser",
    description="upload a pubg video and this app will parse your kill logs and kill/assist/remaining people count with time stamp",
    examples=[["sample.mp4"]],
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
