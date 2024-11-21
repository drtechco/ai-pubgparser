import cv2 
from tqdm import tqdm
from pathlib import Path
import uuid 
import os


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

TARGET_FOLDER="/media/hbdesk/UNTITLED/test_sets/frames/"

all_images = get_file_by_extension(TARGET_FOLDER, (".jpeg", ".jpg", ".png")) 
for image_path in tqdm(all_images):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    cropped_frame = img[0:640, width-640:width]
    fname = Path(image_path).stem 
    # print(os.path.join(TARGET_FOLDER, f"640_{fname}.png"))
    cv2.imwrite(os.path.join(TARGET_FOLDER, f"640_{fname}_{uuid.uuid4().hex}.png"), cropped_frame) 
