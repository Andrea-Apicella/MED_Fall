
from statistics import mode

import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder 
from utils.utility_functions import load_images


class VideoSeqGenerator(Sequence):
    def __init__(self, df: pd.DataFrame, seq_len: int, stride: int, batch_size: int, frames_folder: str):
        self.df = df
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.frames_folder = frames_folder
        
        self.n = len(self.X)
        self.count = 1
        self.len = len(self.X)
        self.evaluate = evaluate
        self.n_windows = (self.batch_size - self.seq_len) // self.stride + 1
        self.n_series_labels = self.n_windows * (len(self.y) // self.batch_size)
        self.n_batches = len(self.y) // self.batch_size
        self.series_labels = []
        self.ys_count = 0
        self.get_item_calls = 0

    def __len__(self):
        return self.X.shape[0] // (self.batch_size)

    def __get_video_seq(self, batch):
        # generate one time series of seq_len padded if its too short
        # check that the time series is from the one cam only (does not overflow on another camera)
        
        #X = load_images(self.frames_folder, batch["frames"])
        
        frames_labels = batch["labels"]
        
        le = LabelsEncoder()
        le.fit(labels)
        
        self.class_indices = dict(zip(le.classes_, le.transform(le.classes_)))
        self.classes = le.transform(labels)
        
        frames_labels_encoded = self.classes
        
        #frames_series contains batch_size series of frames. Each series is a np.array of seq_len frames.
        
        frames_seq = np.empty(self.batch_size)
        cams_seq = np.empty_like(frames_seq)
        s = 0
        for i in range(0, self.batch_size * self.seq_len, self.seq_len):
            frames_seq[i] = load_images(frames_folder, batch["frame_name"][i:i+seq_len])
            cams_seq[i] = df["cam"].tolist()
            
            curr_cam = mode(cams_seq)
            if cams_seq[i] != curr_cam
            for c in range(len(cams_seq))
        
            
            
            
            
            
            
        time_series = [np.empty(self.num_features)] 
        y_s = [None] * self.n_windows
        s = 0
        for w in range(0, self.n_windows):
            s = w * self.stride
            features_seq = X[s : s + self.seq_len, :]
            labels_seq = y[s : s + self.seq_len]
            cams_seq = cams[s : s + self.seq_len]
            curr_cam = mode(cams_seq)
            for i, _ in enumerate(cams_seq):
                if cams_seq[i] != curr_cam:
                    features_seq[i] = np.zeros(self.num_features)  # padding
                    labels_seq[i] = -10  # padding
            time_series[w] = features_seq
            # convert time-step labels in one label per time-series
            labels_seq = [l for l in labels_seq if l != -10]
            label = mode(labels_seq)  # label with most occurrence
            y_s[w] = label

        if not self.evaluate:
            self.series_labels.extend(y_s)
            self.ys_count += len(y_s)
        return np.array(time_series), self.labels_encoder.fit_transform(y_s)

    def __getitem__(self, index):
        self.get_item_calls += 1
        a = index * self.batch_size * self.seq_len
        b = (index + 1) * self.batch_size * self.seq_len

        batch = {"frames": self.df["frame_name"][a:b], "labels": self.df[a:b], "cams": self.df["cam"]}
        X, y = self.__get_video_seq(batch)
        return X, y

    def __on_epoch_end(self):
        pass
