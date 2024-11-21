import numpy as np
import cv2
import re

def parse_username(input: str):
    _input = input
    if _input[1] == "[" and _input[0] != "[":
        _input = _input[1:]
    _input = _input.replace("]", "] ")
    return _input                                       


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
def bgr2_3grey(image: np.ndarray):
    '''
    converts a 3 channel rgb image to one that is grayscale but with 3 channels 
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.merge([gray_image, gray_image, gray_image])
    return gray_bgr

