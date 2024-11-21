import os 
import cv2 
import numpy as np



class GameLogWatcher:
    def __init__(self):
        self.previous_frame = None 
    def detect_new_logs(self, similarity_threshold: float, frame: np.ndarray):
        """compare two frames, returns similarity"""
        if self.previous_frame is None:
            print("setting first frame")
            _rimg = cv2.resize(frame, (640,640)) 
            self.previous_frame = frame 
            return False, None

        if self.previous_frame is not None:
            _rimg1 = cv2.resize(self.previous_frame, (640, 640))
            _rimg2 = cv2.resize(frame, (640,640))
            crop_prev_img = cv2.cvtColor(_rimg1[30:280, -540:610], cv2.COLOR_BGR2GRAY)
            crop_cur_img = cv2.cvtColor(_rimg2[30:280, -540:610], cv2.COLOR_BGR2GRAY)
            _, thresh1 = cv2.threshold(crop_prev_img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(crop_cur_img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            diff_img = cv2.absdiff(thresh1, thresh2)
            change_percentage = np.count_nonzero(diff_img) / diff_img.size
            return change_percentage > similarity_threshold, change_percentage 
