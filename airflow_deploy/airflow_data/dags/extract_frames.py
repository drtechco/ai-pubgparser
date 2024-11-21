from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import os
import uuid
import subprocess

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

def extract_frames_from_video(target_folder: str, target_export_folder: str):
    all_videos = get_file_by_extension(target_folder, (".mp4"))
    print(f"found videos: {all_videos}")
    if not os.path.exists(target_export_folder):
        os.makedirs(target_export_folder)

    for _idx, _a in enumerate(all_videos):
        ffmpeg_command = ["ffmpeg", "-i" ,f"{_a}" , "-vf","fps=0.5" ,f"{target_export_folder}/1080_{uuid.uuid4().hex}_%08d.png"]
        with subprocess.Popen(ffmpeg_command) as ffmpeg_proc:
            try: 
                ffmpeg_proc.communicate()
            except KeyboardInterrupt:
                print("converion stopped")
            finally:
                ffmpeg_proc.terminate()



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG("extract_frames_from_video") as _dag:
    extract_ = PythonOperator(
            task_id="extract_frames_from_raw_video",
            python_callable=extract_frames_from_video,
            op_kwargs={
                'target_folder': "/raw_video_data",
                'target_export_folder': os.path.join("/frames_extract", str(uuid.uuid4().hex))
                },
            )

extract_
