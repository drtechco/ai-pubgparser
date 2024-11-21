'''
USAGE:
    - install requirements with requirements_labelwserver.txt
    - run `python3 label_with_server.py --help` for options
'''

import json 
import argparse 
import os 
import requests
import base64
from tqdm import tqdm

def convertImg2b64(imgpath: str):
    with open(imgpath, "rb") as img_file:
        imgstring = base64.b64encode(img_file.read())
    return imgstring

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



def get_labelmeJSON(filename: str, confidence_threshold: float):
    with open(filename, "rb") as imgfile:
        files = {
                "file": (os.path.basename(filename), imgfile, "image/jpeg")
                }
        data = {
                "confidence_thresh": confidence_threshold
                }
        response = requests.post(opts.server, files=files, data=data)
        if response.status_code == 200:
            annotation_data = response.json()
            return annotation_data
        else:
            return None

def save_annotation(image_path: str, annotation: dict):
    json_filename = os.path.splitext(image_path)[0] + ".json"
    annotation["imageData"] = convertImg2b64(image_path).decode("utf-8")
    _a = json.dumps(annotation, indent=2)
    with open(json_filename, "w") as writejson:
        writejson.write(_a)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True,type=str, help="autolabel server address with route\ne.g http://localhost:3000/label_ui_lm")
    parser.add_argument("--src", required=True, type=str, help="folder with images")
    parser.add_argument("--conf", required=False, default=0.85, help="confidence threshold")
    opts = parser.parse_args() 
    all_imgs = get_file_by_extension(opts.src, (".jpeg", ".png", ".jpg")) 
    for img in tqdm(all_imgs):
        annotation = get_labelmeJSON(img, opts.conf)
        if annotation is not None:
            save_annotation(img, annotation["results"])

