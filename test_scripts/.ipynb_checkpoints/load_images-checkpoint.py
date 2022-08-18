
from statistics import mode

import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder 
from utils.utility_functions import load_images

frames_folder =  "/home/jovyan/work/MED_Fall/vision/vision_dataset/extracted_frames"

def get_video_seq(batch):
    
    frames_tensor = load_images(frames_folder, batch)
    return frames_tensor
    







