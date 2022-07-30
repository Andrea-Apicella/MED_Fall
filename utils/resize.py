import cv2
import numpy as np
import tensorflow as tf
from tf.keras.preprocessing.image import smart_resize
from tqdm.auto import tqdm

from utils.utility_functions import listdir_nohidden_sorted as lsdir


def resize(videos_folder: str, imgs_dest: str, size: tuple):
    videos_paths = lsdir(videos_folder)
    for _, folder in enumerate((t0 := tqdm(videos_paths, position=0))):
        videos = lsdir(f"{folder}/videos")

        for _, cam in enumerate((t1 := tqdm(videos, position=1))):
            start = cam.rfind("/") + 1
            cap = cv2.VideoCapture(cam)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(f"{cam[start:]}.mp4", fourcc, 25, (1920 / 3, 1080 / 3))
            try:
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in range(n_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = smart_resize(frame, size)
                    out.write(frame)
                    cv2.imwrite(imgs_dest, frame)
            finally:
                cap.release()
                out.release()
