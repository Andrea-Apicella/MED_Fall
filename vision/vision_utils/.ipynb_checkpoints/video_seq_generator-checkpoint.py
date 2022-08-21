
from statistics import mode

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder 
from utils.utility_functions import load_images, listdir_nohidden_sorted as lsdir


class VideoSeqGenerator(Sequence):
    def __init__(self, df: pd.DataFrame, seq_len: int, batch_size: int, frames_path: str, input_shape: tuple[int, int, int] = (224, 224, 3)):
        self.df = df,
        self.df = self.df[0]
        self.frames_path: str = frames_path,
        self.seq_len: int = seq_len,
        self.batch_size: int = batch_size,
        self.input_shape: tuple = input_shape,
        
    def __get_cam(self, name: str) -> int
        start = name.find("cam_")
        ind = start + 4
        cam_number = name[ind]
        return int(cam_number)
    
    def __len__(self):
        print(type(self.batch_size), type(self.seq_len))
        length: int = self.df.shape[0] // (self.batch_size * self.seq_len)
        return length
    
    
    def get_video_seq(self, df) -> tuple[np.array, np.array]:
        frames: list = df["frame_name"].tolist()

        # add cam column to dataframe
        cams: list = [get_cam(cam) for cam in frames]

        labels: list = df["macro_labels"].tolist()
        le = LabelEncoder()
        labels_encoded: list = le.fit_transform(labels).tolist()

        X = []
        y = []
        for i in range(0, self.batch_size):
            s: int = i * seq_len
            frames_seq: list = frames[s: s + seq_len]  # select seq_len frame_names
            labels_seq: list = labels_encoded[s: s + seq_len]  # select seq_len labels
            cams_seq: list = cams[s: s + seq_len]  # select seq_len cams

            # check that cam is
            sequence_cam = mode(cams_seq)

            for j in range(len(labels_seq)):
                curr_cam = cams[i]
                if curr_cam != sequence_cam:
                    labels_seq[j] = -10
                    frames_seq[j] = -10

            labels_seq = [label for label in labels_seq if not label == -10]
            frames_seq = [frame for frame in frames_seq if not frame == -10]

            sequence_label = mode(labels_seq)

            images = load_images(self.frames_path, frames_seq, extension="jpg")

            n_seq_frames = np.uint8(seq_len)
            n_actual_frames = np.uint8(images.shape[0])

            n_missing: np.uint8 = n_seq_frames - n_actual_frames

            if n_missing:
                to_add_shape = (n_missing,) + input_shape
                to_add = np.zeros(shape=to_add_shape, dtype=np.uint8)
                images = np.concatenate((images, to_add), axis=0)


            X.append(images)
            y.append(sequence_label)

        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)

        return X, y


    def __get_item__(self, index):
        start: int = index * self.batch_size * self.seq_len
        end: int = (index + 1) * self.batch_size * self.seq_len
        
        X, y = self.get_video_seq(pd.DataFrame(self.df.iloc[start:end, :]))
                                  
        return X, y
                                  
    def __on_epoch_end(self):
        pass
        
        
        
        
        
      