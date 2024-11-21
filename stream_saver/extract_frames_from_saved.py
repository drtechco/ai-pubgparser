import os 
import subprocess 
import argparse 
import datetime
import uuid

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

TARGET_FOLDER="/media/hbdesk/UNTITLED/test_sets"

all_mp4s = get_file_by_extension(TARGET_FOLDER, (".mp4"))

for _idx, _a in enumerate(all_mp4s):
    # if not os.path.exists(f"saved_640o/{_idx}"):
    #     os.makedirs(f"saved_640o/{_idx}")
    ffmpeg_command = ["ffmpeg", "-i" ,f"{_a}" , "-vf","fps=1" ,f"/media/hbdesk/UNTITLED/test_sets/frames/1080_{uuid.uuid4().hex}_%08d.png"]
    with subprocess.Popen(ffmpeg_command) as ffmpeg_proc:
        try: 
            ffmpeg_proc.communicate()
        except KeyboardInterrupt:
            print("convert stopped")
        finally:
            ffmpeg_proc.terminate()
