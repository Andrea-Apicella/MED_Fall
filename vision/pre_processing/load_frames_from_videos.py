from utils import listdir_nohidden_sorted
from tqdm.auto import tqdm, trange
import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf

class FramesExtractor:
    def __init__(self, videos_folder, labels, ground_truth_folder, frame_size=(224,224)):
        self.videos_folder = videos_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)
        self.labels = labels
        self.ground_truth_folder = ground_truth_folder
        self.frame_size=frame_size

    def extract_frames(self):
        extracted_frames = []

        ground_truth = pd.DataFrame()
        for _, folder in enumerate(self.videos_paths):
            folder_name = folder.replace(self.videos_folder, "")[1:]
            # t0.set_description(f"Processing folder: {folder_name}")

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
            for _, cam in enumerate((t1 := tqdm(videos))):
                start = cam.rfind("/") + 1
                end = len(cam) - 4
                t1.set_description(f'Extracting frames from: {cam[start:].replace(" ", "_")}')
                cap = cv2.VideoCapture(cam)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for f in trange(n_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = tf.keras.preprocessing.image.smart_resize(
                            frame, self.frame_size)
                        extracted_frames.append(frame)
                        # features = self.predict_frame(frame)
                        # folder_features.append(features)
                        #
                        # file_name = f'{cam[start:end].lower().replace(" ", "_")}_{str(f).zfill(4)}'
                        # frames_names.append(file_name)

                finally:
                    cap.release()

            ground_truth = pd.concat([ground_truth,folder_ground_truth], axis=0, ignore_index=True)
            ground_truth.drop(columns=ground_truth.columns[0],
                    axis=1,
                    inplace=True)
        return np.array(extracted_frames), ground_truth

#%%
