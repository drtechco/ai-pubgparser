import cv2 
import os 
import numpy as np



def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate image entropy as a measure of information content.
    """
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

def get_metrics(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    mean_value = np.mean(gray)
    entropy = calculate_entropy(gray)
    metrics = {
        'variance': variance,
        'laplacian_variance': laplacian_var,
        'mean_value': mean_value,
        'entropy': entropy
    }
    return metrics
