import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm, trange
from utils.utility_functions import listdir_nohidden_sorted, safe_mkdir


projectdir = 'home/jovyan/work/MED_Fall'

class FeaturesExtractor:
    def __init__(self, videos_folder, features_extractor, preprocess_input):
        self.videos_folder = videos_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)
        self.features_extractor = features_extractor
        self.preprocess_input = preprocess_input

    def predict_frame(self, frame):
        size = self.features_extractor.input_shape[1:3]
        frame = frame = tf.keras.preprocessing.image.smart_resize(
            frame, size)
        frame = self.preprocess_input(frame)
        return self.features_extractor.predict(np.expand_dims(frame, axis=0))

    def extract_features(self, dest):
        
        safe_mkdir(dest)
        all_features = []
        for _, folder in enumerate((t0 := tqdm(self.videos_paths, position=0, leave=False))):
            folder_name = folder.replace(self.videos_folder, "")[1:]
            t0.set_description(
                f'Processing folder: {folder_name}')

            sheet_name = folder.replace(self.videos_folder, "").lower()[1:]
            try:
                labels_sheet = pd.read_excel(f"{projectdir}/dataset/labels.xlsx",
                                             sheet_name=sheet_name, index_col=0)
                os.path.exists(
                    f"{projectdir}/vision/vision_dataset/ground_truth/{folder_name}.csv")

            except OSError:
                print(
                    f'Labels sheet {sheet_name} not found. Skippig {folder_name}.')
                continue

            #if os.path.exists(f"outputs/dataset/dataset/{folder_name}.csv"):
                #print(
                    #f'Folder already processed. Skippig {folder_name}')
                #continue

            video_iso_files_path = f'{folder}'

            frames_names = []
            folder_features = []

            video_iso_files = listdir_nohidden_sorted(
                video_iso_files_path)

            for _, cam in enumerate((t1 := tqdm(video_iso_files, leave=False))):
                start = cam.rfind('/') + 1
                end = len(cam) - 4
                t1.set_description(
                    f'Extracting frames from: {cam[start:].replace(" ", "_")}')
                cap = cv2.VideoCapture(cam)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for f in trange(n_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        features = self.predict_frame(frame)
                        folder_features.append(features)

                        file_name = f'{cam[start:end].lower().replace(" ","_")}_{str(f).zfill(4)}'
                        frames_names.append(file_name)

                finally:
                    cap.release()

            
            # save compressed features as npz files
            folder_features = np.asarray(folder_features).squeeze()
            np.savez_compressed(
                f"{dest}/{folder_name}", folder_features)

            print(f"folder_features shape: {folder_features.shape}")
        
        
if __name__ == "__main__":
    projectdir = '/home/jovyan/work/MED_Fall'
    
    videos_folder = f'{projectdir}/vision/vision_dataset/DATASET_WP8_resized'
    features_extractor_path = f'{projectdir}/vision/models/vgg_feature_extractor.h5'
    features_extractor = tf.keras.models.load_model(features_extractor_path)
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    fe = FeaturesExtractor(videos_folder=videos_folder, features_extractor=features_extractor, preprocess_input=preprocess_input)
    
    fe.extract_features(dest=f'{projectdir}/vision/vision_dataset/vgg_features')
