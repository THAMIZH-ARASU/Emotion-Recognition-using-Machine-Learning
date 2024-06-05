import cv2
import os
import numpy as np

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (48, 48))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(images):
    images = images / 255.0  # Normalize pixel values
    return images.reshape(images.shape[0], -1)  # Flatten images
