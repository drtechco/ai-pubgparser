from pixel_filter import PixelClassifier 
import numpy as np
import uuid
from ultralytics import YOLO
import cv2 
import argparse
import time
_a = PixelClassifier("./pixel_filter.pth")



def parse_username(input: str):
    _input = input
    if _input[1] == "[" and _input[0] != "[":
        _input = _input[1:]
    _input = _input.replace("]", "] ")
    return _input                                       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="path to testing image")
    opts = parser.parse_args()
    test_img = cv2.imread(opts.img)
    char_class, debug_img = _a.get_char_class(test_img, debug=True, confidence_threshold=0.85)
    print("PIXEL, CLASS: ", char_class)
