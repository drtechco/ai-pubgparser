import torch 
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision import models
import torch.nn as nn

class CharClassifier:
    def __init__(self, model_path, inference_size: int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = self.cc_load_classes("./models/char_list.txt")
        self.model = self.cc_load_model(model_path, len(self.classes)) #hard coded for now. 
        self.inference_size = inference_size
    def cc_load_classes(self, class_path):
        with open(class_path, "r") as file:
            items = [line.strip() for line in file]
        return items
    def cc_load_model(self, model_path, num_classes):
        model = models.resnet34(pretrained=True)
        # model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.eval()
    
    def cc_preprocess_image_np(self, input_image: np.ndarray):
        h, w ,c = input_image.shape
        pad_width = max(0, 24 - w)
        pad_height = max(0, 24 - h)
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        _pad = np.pad(input_image, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0)        
        image_resize = cv2.resize(_pad, (self.inference_size,self.inference_size))
        image_resized = image_resize.astype(np.float32) / 255.0
        # image_resized = bgr2_3grey(image_resized)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_resized - mean) / std
        input_tensor = np.transpose(image_normalized, (2, 0, 1))
        return torch.from_numpy(input_tensor).unsqueeze(0).float()

    def cc_preprocess_image(self, image_path: str):
        preprocess = transforms.Compose([
            transforms.Resize((self.inference_size,self.inference_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        return input_tensor.unsqueeze(0)
    
    def cc_predict_weapon(self, input, confidence_threshold: float = 0.7):
        """
        returns predicted_class, confidence
        """
        with torch.no_grad():
            output = self.model(input)
            probabilities = nn.Softmax()(output)
            _ , predicted_idx = torch.max(output, 1)
        predicted_class = self.classes[predicted_idx.item()]
        return predicted_class, probabilities[0][predicted_idx]
        

    def get_char_class(self, input_image: np.ndarray | str, confidence_threshold: float = 0.5, debug: bool = False):
        assert(confidence_threshold > 0.0 and confidence_threshold < 1.0)
        if type(input_image) == np.ndarray: 
            input_batch = self.cc_preprocess_image_np(input_image)
        elif type(input_image) is str and os.path.exists(input_image):
            input_batch = self.cc_preprocess_image(input_image)
        else:
            return 
        predictions, confidence = self.cc_predict_weapon(input_batch, confidence_threshold)
        print(f"[DEBUG] char_prediction: {predictions} confidence: {confidence}")
        if confidence < confidence_threshold:
            if debug:
                _debug = input_batch.squeeze(0).float()
                _debug = np.transpose(_debug, (1, 2, 0))  # Shape now (112, 112, 45)
                _debug = _debug * 255
                return "", _debug 
            else:
                return ""
             
        else:
            if not debug:
                return predictions
            else: 
                _debug = input_batch.squeeze(0).float()
                _debug = np.transpose(_debug, (1, 2, 0))  # shape now (112, 112, 45)
                _debug = _debug * 255
                return predictions, _debug 


def bgr2_3grey(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.merge([gray_image, gray_image, gray_image])
    return gray_bgr
