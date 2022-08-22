
from statistics import mode

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder 
from utils.utility_functions import listdir_nohidden_sorted as lsdir
import sys
import cv2

class FeatureSeqGenerator(Sequence):
    def __init__(self, model, features_shape: int, df: pd.DataFrame, seq_len: int, batch_size: int, frames_path: str, label_encoder, input_shape: tuple[int, int, int] = (224, 224, 3)):
        self.model = model
        self.features_shape = features_shape
        self.df = df,
        self.df = self.df[0]
        self.label_encoder = label_encoder
        self.frames_path: str = frames_path,
        self.frames_path = self.frames_path[0]
        self.seq_len: int = seq_len,
        self.seq_len = self.seq_len[0]
        self.batch_size: int = batch_size,
        self.batch_size = self.batch_size[0]
        #print("self.batch_size type: ", type(self.batch_size))
        #print("self.seq_len type: ", type(self.seq_len))
        self.input_shape: tuple = input_shape,
        self.input_shape = self.input_shape[0]
        
        
        cams = []
        for name in self.df["frame_name"]:
            cam = self.__get_cam(name)
            cams.append(cam)
        self.df["cam"] = cams
        
        actor_sequences = []
        for name in self.df["frame_name"]:
            actor_seq = self.__get_actor_seq(name)
            actor_sequences.append(actor_seq)
        self.df["actor_seq"] = actor_sequences

        
    def __get_cam(self, name: str) -> int:
        start = name.find("cam_")
        ind = start + 4
        cam_number = name[ind]
        return int(cam_number)
    
    def __get_actor_seq(self, name: str) -> str:
        'Actor_1_bed_cam_1_0000'
        start = name.rfind("actor_") + 6
        end = name.rfind("_cam")
        
        actor_seq = name[start:end]
        return actor_seq
        

    
    def __len__(self):
        length: int = len(self.df) // (self.batch_size * self.seq_len)
        #print("length: ", length)
        return length
    
    
    def __extract_features(self,images_dir: str, name: str, resize_shape: tuple[int,int] = (224,224), extension = None) -> np.array:
        """Loads image, rescales and resizes it. Extract features using provided feature extractor 

        Parameters
        ----------
        images_dir: str. Path pointing to the folder that contains the images.

        images_names: list containing the titles of the images.

        """


        if extension is None:
            image = cv2.imread(f"{images_dir}/{name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, resize_shape)
            image = image.astype(float) / 255
            #image = np.expand_dims(image, axis=0)
            features = self.model.predict(image)
            return features
        else: 
            image = cv2.imread(f"{images_dir}/{name}.{extension}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, resize_shape)
            image = image.astype(float) / 255
            image = np.expand_dims(image, axis=0)
            features = self.model.predict(image)
            return features


    
    def get_video_seq(self, df) -> tuple[np.array, np.array]:
        #print(df.iloc[-40:, :])
        
        frames: list = df["frame_name"].tolist()

        # add cam column to dataframe
        #cams: list = [self.__get_cam(name) for name in frames]
        #self.df["cam"] = pd.Series(cam)

        labels: list = df["macro_labels"].tolist()
        cams: list = df["cam"].tolist()
        actors = df["actor_seq"].tolist()
        

        X = []
        y = []
        for i in range(0, self.batch_size):
            
            s: int = i * self.seq_len
            frames_seq: list = frames[s: s + self.seq_len]  # select seq_len frame_names
            labels_seq: list = labels[s: s + self.seq_len]  # select seq_len labels
            cams_seq: list = cams[s: s + self.seq_len]  # select seq_len cams
            actors_seq: list = actors[s: s + self.seq_len]
            
            
            sequence_label = mode(labels_seq)
            sequence_cam = mode(cams_seq)
            sequence_actor_seq = mode(actors_seq)
            
            l = []

            for j in range(self.seq_len):
                curr_cam = cams_seq[j]
                curr_actor = actors_seq[j]
                curr_label = labels_seq[j]

                if curr_label != sequence_label or curr_cam != sequence_cam or curr_actor != sequence_actor_seq:
                    labels_seq[j] = None
                    frames_seq[j] = None
                else: 
                    l.append(labels_seq[j])
            
            labels_seq = l
                    

            
            assert len(labels_seq) > 0
                
            sequence_label = mode(l)
            sequence_label = self.label_encoder.transform([sequence_label]).tolist()[0]
            
            
            features_list = []
            for name in frames_seq:
                if name == None:
                    padding = np.zeros(shape=self.features_shape)
                    features_list.append(padding)
                else:
                    frame_features = self.__extract_features(self.frames_path, name, extension='jpg')
                    frame_features = np.squeeze(frame_features, axis=0)
                    features_list.append(frame_features)


            X.append(features_list)
            y.append(sequence_label)

        X = np.array(X)
        y = np.array(y, dtype=np.uint8)

        return X, y
    
     
    
    
    def test(self):
        series_labels = []
        series_cams = []
        n_missing = 0
        for i in range(0, len(self.df), self.seq_len):
            df = self.df.iloc[i:i+self.seq_len, :]
            
            frames: list = df["frame_name"].tolist()

            # add cam column to dataframe

            labels: list = df["macro_labels"].tolist()
            
            cams: list = df["cam"].tolist()
            
            seq_label = mode(labels)
            seq_cam = mode(cams)
            
            
            for j in range(len(cams)):
                #print("sequence_cam: ", sequence_cam)
                curr_cam = cams[j]
                #print("curr_cam: ", curr_cam)

                if curr_cam != seq_cam:
                    n_missing+=1
                    labels[j] = None
                    frames[j] = None
            
            l = []
            f = []
            for label in labels:
                  if label is not None:
                      l.append(label)
            for frame in frames:
                  if frame is not None:
                      f.append(frame)
                  
            labels_seq = l
            frames_seq = f
            
            
            series_labels.append(seq_label)
            series_cams.append(seq_cam)
        self.classes_ = series_labels
        self.cams_ = series_cams
        self.n_missing = n_missing
        return series_labels, series_cams
        
        


    def __getitem__(self, index):
        self.index = index
        start: int = index * (self.batch_size * self.seq_len)
        end: int = (index + 1) * (self.batch_size * self.seq_len)
        
        X, y = self.get_video_seq(pd.DataFrame(self.df.iloc[start:end, :]))
                                  
        return X, y
                                  
    def __on_epoch_end(self):
        pass
        
        
        
        
        
      