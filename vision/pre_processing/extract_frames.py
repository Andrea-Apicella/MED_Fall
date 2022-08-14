import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from tqdm.auto import tqdm, trange
from utils.utility_functions import listdir_nohidden_sorted, safe_mkdir


class FramesExtractor:
    def __init__(self, videos_folder: str, output_folder: str, labels: str, ground_truth_folder: str, frame_size: tuple[int, int] = (224, 224)) -> pd.DataFrame:
        self.videos_folder = videos_folder
        self.output_folder = output_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)
        self.labels = labels
        self.ground_truth_folder = ground_truth_folder
        self.frame_size = frame_size
        self.available_sequences = len(listdir_nohidden_sorted("/home/jovyan/work/MED_Fall/vision/vision_dataset/ground_truth"))
        print('Number of available sequences: ', self.available_sequences)

    def extract_frames(self):
        extracted_frames = []

        ground_truth = pd.DataFrame()
        for _, folder in enumerate(t0 := tqdm(self.videos_paths[:self.available_sequences])):
            folder_name = folder.replace(self.videos_folder, "")[1:]
            t0.set_description(f"Processing folder: {folder_name}")

            sheet_name = folder.replace(self.videos_folder, "").lower()[1:]
            try:
                labels_sheet = pd.read_excel(self.labels, sheet_name=sheet_name, index_col=0)
                os.path.exists(f"{self.ground_truth_folder}/{folder_name}.csv")

            except OSError:
                print(f"Labels sheet {sheet_name} not found. Skippig {folder_name}.")
                continue

            if os.path.exists(f"self.ground_truth_folder/{folder_name}.csv"):
                print(f"Folder already processed. Skippig {folder_name}")
                continue

            videos_path = f"{folder}/videos"

            folder_ground_truth = pd.read_csv(f"{self.ground_truth_folder}/{folder_name}.csv")

            videos = listdir_nohidden_sorted(videos_path)
            folder_frames = []
            for _, cam in enumerate((t1 := tqdm(videos, leave=False))):
                start = cam.rfind("/") + 1
                end = len(cam) - 4
                t1.set_description(f'Extracting frames from: {cam[start:].replace(" ", "_")}')
                cap = cv2.VideoCapture(cam)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for f in trange(n_frames, leave=False):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = tf.keras.preprocessing.image.smart_resize(frame, self.frame_size)
                        frame_name = f"{cam[start:end].replace(' ', '_')}_{str(f).zfill(4)}.jpg"
                        frame_name = frame_name.lower()
                        frame_path = f"{self.output_folder}/{frame_name}"
                    
                        cv2.imwrite(frame_path, frame)

                finally:
                    cap.release()
                    

if __name__ == "__main__":
    
    projectdir = "/home/jovyan/work/MED_Fall/"
    
    fe = FramesExtractor(
        videos_folder       = f"{projectdir}/vision/vision_dataset/DATASET_WP8_resized/",
        output_folder       = f"{projectdir}/vision/vision_dataset/extracted_frames/",
        labels              = f"{projectdir}/dataset/labels.xlsx",
        ground_truth_folder = f"{projectdir}/vision/vision_dataset/ground_truth/",
    )
    
    fe.extract_frames()
    
    extracted_frames_persistent = '/home/jovyan/work/persistent/extracted_frames/'
    dir_created = safe_mkdir(extracted_frames_persitent)
    if dir_created:
        shutil.copy(fe.output_folder, extracted_frames_persistent)
