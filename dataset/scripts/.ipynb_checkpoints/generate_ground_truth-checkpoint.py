import os

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm, trange
from utils.utility_functions import listdir_nohidden_sorted as lsdir



projectdir = '/home/jovyan/work/MED_Fall'

def generate_ground_truth(videos_folder, output_folder):
    
    videos_paths = lsdir(videos_folder)
    

    

    for _, folder in enumerate((t0 := tqdm(videos_paths, position=0))):
                folder_name = folder.replace(videos_folder, "")[1:]
                t0.set_description(
                    f'Processing folder: {folder_name}')

                sheet_name = folder.replace(videos_folder, "").lower()[1:]
                try:
                    labels_sheet = pd.read_excel(f"{projectdir}/dataset/labels.xlsx",
                                                 sheet_name=sheet_name, index_col=0)
                    os.path.exists(
                        f"{projectdir}/vision/vision_dataset/ground_truth/{folder_name}.csv")

                except OSError:
                    print(
                        f'Labels sheet {sheet_name} not found. Skippig {folder_name}.')
                    continue
                video_iso_files = lsdir(folder)
                    
                df = pd.concat([labels_sheet] * len(video_iso_files), ignore_index=True) 
                df.to_csv(f"{output_folder}/{folder_name}.csv")
                
                
if __name__ == "__main__":
    
    
    generate_ground_truth(videos_folder=f"{projectdir}/vision/vision_dataset/DATASET_WP8_resized", output_folder = f"{projectdir}/vision/vision_dataset/ground_truth")