import os
import cv2
def create_directories(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

