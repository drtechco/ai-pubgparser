import torch 
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision import models
import torch.nn as nn

class WeaponsClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = self.wc_load_classes("../weapons_classifier/weapons_list.txt")
        self.model = self.wc_load_model(model_path, len(self.classes)) #hard coded for now. 
    def wc_load_classes(self, class_path):
        with open(class_path, "r") as file:
            items = [line.strip() for line in file]
        return items
    def wc_load_model(self, model_path, num_classes):
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.eval()
    
    def wc_preprocess_image_np(self, input_image: np.ndarray):
        image_resized = cv2.resize(input_image, (224,224))
        image_resized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_resized - mean) / std
        input_tensor = np.transpose(image_normalized, (2, 0, 1))
        return torch.from_numpy(input_tensor).unsqueeze(0).float()

    def wc_preprocess_image(self, image_path: str):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        return input_tensor.unsqueeze(0)
    
    def wc_predict_weapon(self, input):
        with torch.no_grad():
            output = self.model(input)
            _ , predicted_idx = torch.max(output, 1)
        predicted_class = self.classes[predicted_idx.item()]
        return predicted_class
        

    def get_weapon_class(self, input_image: np.ndarray | str):
        if type(input_image) == np.ndarray: 
            input_batch = self.wc_preprocess_image_np(input_image)
        elif type(input_image) is str and os.path.exists(input_image):
            input_batch = self.wc_preprocess_image(input_image)
        else:
            return 
        predictions = self.wc_predict_weapon(input_batch)
        return predictions
