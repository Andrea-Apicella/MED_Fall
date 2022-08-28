from statistics import mode

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


class VideoSeqGenerator(Sequence):
    """
    VideoSeqGenerator is a custom Keras Generator that produces windows (time-series) of video frames.
    ------

    arguments:
        - frames_path: str: the path to the folder which contains the video frames as jpg image files.
        - df: dataframe containing a column with the names of the frames and a column with the corresponding labels.
        - seq_len: int: window size. The number of frames to insert in a single time-series.
        - batch_size: int: the number of windows to put in a batch.
        - label_encoder: a sklearn.preprocessing encoder object to transform class labels into OneHot vectors or categorical labels.
        - input_shape: tuple. Optional. The desired shape (resolution and num of channels) for a single image. Default is (224,224,3).

    outputs:
        - X: np.array: batch of windows to be fed to a model. X has shape (batch_size, seq_len, input_shape)
        - y: np.array: batch of windows' encoded labels. y has shape (batch_size, num_classes) if the provided encoder is a OneHotEncoder.
    
    """

    def __init__(self, df: pd.DataFrame, seq_len: int, batch_size: int, stride:int, frames_path: str, label_encoder,
                 input_shape: tuple[int, int, int] = (224, 224, 3)):
        self.df = df,
        self.df = self.df[0]
        self.label_encoder = label_encoder
        self.frames_path: str = frames_path,
        self.frames_path = self.frames_path[0]
        self.seq_len: int = seq_len,
        self.seq_len = self.seq_len[0]
        self.batch_size: int = batch_size,
        self.batch_size = self.batch_size[0]
        self.stride = stride
        self.input_shape: tuple = input_shape
        print(self.input_shape)
        print(type(self.input_shape))
        #self.input_shape = self.input_shape[0]
        
        labels: list = df["macro_labels"].tolist()  # extract labels as list from df

        y_series = []  # instantiate empty list that will contain windows' labels
        
        s_s = []
        for i in range(0, len(self.df) // (self.batch_size*self.seq_len)):  # iterate over the batch_size
            
            s: int = i * self.stride  # update stride (this splitting in time-series operation strides of seq_len each time).
            s_s.append(s)
        
            labels_seq: list = labels[s: s + self.seq_len]  # select seq_len labels from df

            sequence_label = mode(labels_seq)  # calculate the mode of the frames' labels
            y_series.append(sequence_label)
        
        self.windows_labels = y_series
        self.s_s = s_s
        # add to the input dataframe a column with the cam number of each frame. If frame_name is "actor_1_bed_cam_1_0000", cam number is int("1").
        cams = []
        for name in self.df["frame_name"]:
            cam = self.__get_cam(name)
            cams.append(cam)
        self.df["cam"] = cams

        # add to the input dataframe a column with the actor_sequence string of each frame. If frame_name is "actor_1_bed_cam_1_0000", actor_seq is "actor_1_bed".
        actor_sequences = []
        for name in self.df["frame_name"]:
            actor_seq = self.__get_actor_seq(name)
            actor_sequences.append(actor_seq)
        self.df["actor_seq"] = actor_sequences

    def __get_cam(self, name: str) -> int:
        """
        Extracts the cam number from a frame_name.
        If frame_name is "actor_1_bed_cam_1_0000", cam number is int("1").
        """
        start = name.find("cam_")
        ind = start + 4
        cam_number = name[ind]
        return int(cam_number)

    def __get_actor_seq(self, name: str) -> str:
        """
        Extracts the actor and the sequence from a frame_name.
        If frame_name is "actor_1_bed_cam_1_0000", actor_seq is "actor_1_bed".
        """
        start = name.rfind("actor_") + 6
        end = name.rfind("_cam")

        actor_seq = name[start:end]
        return actor_seq

    def __len__(self):
        """
        Calculates the number of training steps by dividing the length of the input by the number of frames in a batch
        (The number of frame in a batch is batch_size * seq_len).
        """
        length: int = len(self.df) // (self.batch_size * self.seq_len)
        return length

    def __load_image(self, images_dir: str, name: str, resize_shape: tuple[int, int], extension=None) -> np.array:
        """Loads a video frame into a np.array, rescales its pixel values in the interval (0, 1), resizes the frame to resize_shape.

        ----------
        arguments:
        
        - images_dir: str: Path pointing to the folder that contains the video frames as jpg image files.
        - name: str: single name of a frame.
        - resize_shape: tuple[int,int]: desired frame resolution.
        - extension: extension to append to the frame name if not present. If no parameter is provided the default appended extension will be '.jpg'.
        
        output:
        image: np.array: loaded frame with shape resize_shape.
        """

        if extension is None:
            image = cv2.imread(f"{images_dir}/{name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_shape)
            #image = image.astype(float) / 255
            return image
        else:
            image = cv2.imread(f"{images_dir}/{name}.{extension}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_shape)
            #image = image.astype(float) / 255
            return image

    def get_video_seq(self, df) -> tuple[np.array, np.array]:
        """
        
        """

        frames: list = df["frame_name"].tolist()  # extract frames names as list from df
        labels: list = df["macro_labels"].tolist()  # extract labels as list from df
        cams: list = df["cam"].tolist()  # extract cam numbers as list from df
        actors = df["actor_seq"].tolist()  # extract actor sequences as list from df

        X = []  # instantate empty list that will contain loaded windows of frames
        y = []  # instantiate empty list that will contain OneHotEncoded labels

        for i in range(0, self.batch_size):  # iterate over the batch_size
            
            s: int = i * self.stride  # update stride (this splitting in time-series operation strides of seq_len each time).
            frames_seq: list = frames[s: s + self.seq_len]  # select seq_len frame_names from df
            labels_seq: list = labels[s: s + self.seq_len]  # select seq_len labels from df
            cams_seq: list = cams[s: s + self.seq_len]  # select seq_len cams from df
            actors_seq: list = actors[s: s + self.seq_len]  # select seq_len actor sequnces from df

            sequence_label = mode(labels_seq)  # calculate the mode of the frames' labels
            sequence_cam = mode(cams_seq)  # calculate the mode of the cams' numbers
            sequence_actor_seq = mode(actors_seq)  # calculate the mode of the actor sequnces

            l = []  # instantiate empty list that will contain only the labels of the frames to actually load

            for j in range(self.seq_len):  # iterate over the current window
                curr_cam = cams_seq[j]  # select the j-th cam number in the window
                curr_actor = actors_seq[j]  # select the j-th actor sequence in the window
                curr_label = labels_seq[j]  # select the j-th label in the window
                #print("curr_label", curr_label, "sequence_label", sequence_label)

                # if the j-th frame has a different label, or cam number, or actor sequence respect to the corresponding modes of the window
                # replace the j-th label and j-th frame_name with None 
                if curr_label != sequence_label or curr_cam != sequence_cam or curr_actor != sequence_actor_seq:
                    labels_seq[j] = None
                    frames_seq[j] = None
                else:
                    l.append(labels_seq[j])  # else, append the j-th label to l

            labels_seq = l  # overwrite window's labels with l, which only contains the label of the frames that are actually loaded for this window.

            assert len(
                labels_seq) > 0  # make sure that the mode operations have not deleted all the frames from the window.

            assert mode(
                l) == sequence_label  # make sure that after removing the unwanted frames from the window, the mode of the labels is still the same.

            sequence_label = self.label_encoder.transform([sequence_label]).tolist()[
                0]  # OneHotEncode the mode of the labels, a.k.a t window's label.

            images = []  # istantiate empty list that will contain the loaded frames
            for name in frames_seq:  # iterate over window's frames

                if name == None:
                    # add padding image (all pixels equal to 0) if the frame name is None. 
                    # This happens if the curr frame has a different cam or actor_seq from the mode of the window
                    pad_image = np.zeros(shape=self.input_shape, dtype=np.uint8)
                    images.append(pad_image)
                else:
                    # else, load the image and append it to images
                    loaded_image = self.__load_image(self.frames_path, name, extension="jpg",
                                                     resize_shape=self.input_shape[:2])
                    images.append(loaded_image)

            X.append(images)  # append the i-th window of frames to X
            y.append(sequence_label)  # append the i-th window's label to y

        # convert X and y to numpy arrays
        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.uint8)

        return X, y




    def __getitem__(self, index):
        """
        Loads one batch into memory. Keras custom generators must have a __getitem__ method.
        """
        self.index = index  # index is generated from keras during fit. Store it to be able to use __getitem__ to extract a batch for checking it is correctly created.
        start: int = index * (self.batch_size * self.seq_len)
        end: int = (index + 1) * (self.batch_size * self.seq_len)

        # load a batch using get_video_seq function
        X, y = self.get_video_seq(pd.DataFrame(self.df.iloc[start:end, :]))

        return X, y

    def __on_epoch_end(self):
        pass
