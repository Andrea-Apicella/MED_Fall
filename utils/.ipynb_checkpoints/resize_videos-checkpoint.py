import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
from tqdm import tqdm, trange

from utility_functions import listdir_nohidden_sorted as lsdir
from utility_functions import safe_mkdir


def resize(dataset: str, dest_dataset: str, size: tuple):
    videos_paths = lsdir(dataset)
    safe_mkdir(dest_dataset)
    for _, folder in enumerate((t0 := tqdm(videos_paths, position=0))):
        t0.set_description(f"processing folder {folder}")
        videos = lsdir(f"{folder}/videos")
        folder_name_start = folder.rfind("/") + 1
        folder_name = folder[folder_name_start:]
        curr_video_out_folder = f"{dest_dataset}/{folder_name}"
        safe_mkdir(curr_video_out_folder)

        for _, cam in enumerate(tqdm(videos, position=1)):
            start = cam.rfind("/") + 1
            cap = cv2.VideoCapture(cam)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video_path = f"{dest_dataset}/{folder_name}/{cam[start:]}"
            print('resizing into ', output_video_path)
            out = cv2.VideoWriter(output_video_path, fourcc, 30, size)
            try:
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in trange(n_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(src=frame, dsize=size, interpolation=cv2.INTER_AREA)
                    out.write(frame)
            finally:
                cap.release()
                out.release()


if __name__ == "__main__":
    resize(dataset="/home/jovyan/work/persistent/DATASET_WP8", dest_dataset="/home/jovyan/work/MED_Fall/vision/vision_dataset/DATASET_WP8_resized", size=(640, 360))
