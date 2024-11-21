from airflow import DAG
from tqdm import tqdm
import cv2 
import numpy as np
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import uuid
import shutil

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
def crop_image(img: np.ndarray, destination_folder: str, crop_size: int):
    height, width = img.shape[:2]
    cropped_frame = img[0:crop_size, width-crop_size:width]
    cv2.imwrite(os.path.join(destination_folder, f"640_{uuid.uuid4().hex}.jpeg"), cropped_frame)
    return cropped_frame

def preprocess_raw_frames(raw_frames_folder: str, destination_folder: str):
    all_raws = [f.path for f in os.scandir(raw_frames_folder) if f.is_dir()]
    for folder in all_raws:
        all_raw_frames = get_file_by_extension(folder, (".png", ".jpeg", ".jpg"))
        for image_path in tqdm(all_raw_frames):
            img = cv2.imread(image_path)
            # crops 640x640 ui detection image
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
                os.makedirs(os.path.join(destination_folder, "ui"))
            cropped_image = crop_image(img, os.path.join(destination_folder, "ui") , 640)
    for _raws in all_raws:
        print("RAW: ", _raws)
        shutil.rmtree(_raws)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

FOLDER = os.path.join("/preprocessed_data", str(uuid.uuid4().hex))

with DAG("pre_process_raw_frames") as _dag:
    extract_ = PythonOperator(
            task_id="pre_process_raw_frames",
            python_callable=preprocess_raw_frames,
            op_kwargs={
                'raw_frames_folder': "/frames_extract",
                'destination_folder': FOLDER
                },
            )

extract_
