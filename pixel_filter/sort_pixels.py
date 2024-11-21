from pixel_filter import PixelClassifier 
import numpy as np
import uuid
from ultralytics import YOLO
import cv2 
import argparse
import time
import os
_a = PixelClassifier("./pixel_filter.pth")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="path to raw images")
    parser.add_argument("--dest", type=str, help="path to export folder")
    opts = parser.parse_args()
    all_jsons = get_file_by_extension(opts.source, (".jpeg", ".png", ".jpg"))
    for _anno in all_jsons:
        test_img = cv2.imread(os.path.join(opts.source,_anno))
        char_class, debug_img = _a.get_char_class(test_img, debug=True, confidence_threshold=0.85)
        if char_class == "clear" or char_class == "not_clear":
            src_dir = os.path.join(opts.source, _anno)
            _subf = os.path.join(opts.dest, char_class)
            fname = os.path.basename(_anno)
            dest_dir = f"{_subf}/{fname}"  
            # print(src_dir)
            # print(dest_dir)
            os.rename(src_dir, dest_dir)
        else:
            continue
