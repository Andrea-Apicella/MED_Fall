from collections import Counter
from statistics import mode

import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

from utils.utility_functions import listdir_nohidden_sorted as lsdir

class DatasetLoader:
    """
    Class that stores the specs needed to load features extracted from single frames.
    ---
    
    Parameters:
    - dataset_folder: str. Path to folder containing frames names and labels csv files.
    - features_folder: str. Path to folder containing the npz compressed features files.
    - actors: list. List containing the actor numbers to actually load. Example: actors = [1, 2, 3] will load the first three actors' features.
    - cams: list. List containing the cameras to load for each actor. Example: cams = [5, 6, 7] will load the last three cameras of each specified actor.
    - drop_offair: bool. If true the off_air frames will not be loaded.
    """
    
    def __init__(self, dataset_folder: str, features_folder: str, actors: list, cams: list, drop_offair: bool):
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.actors = actors
        self.cams = cams
        self.drop_offair = drop_offair

    def load(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Method that actually loads the frames' features into memory.
        -----
        
        Outputs:
        - dataset: pd.DataFrame: dataframe containing the frame names and corresponding labels of the specified actors and respective cameras.
        - all_features: np.array. numpy array containing the frames' features of the specified actors and respective cameras.
        """
        
        datasets_paths = lsdir(self.dataset_folder) # list all the csv files.
        features_paths = lsdir(self.features_folder) # list all the npz features files.
        
        # remove unwanted actors from the dataset and features lists.
        if self.actors:
            self.actors = [int(a) for a in self.actors] # convert user input list to list of intergers
            
            indices = [] # empty list that will store the indices of the actors to NOT load
            
            for i, filename in enumerate(datasets_paths): # for each csv file name
                if int(filename[filename.find("Actor_") + 6]) not in self.actors: #if the actor number in the file name is not in self.actors
                    indices.append(i) # append to indices the index to remove

            for index in sorted(indices, reverse=True): #sort descending the indices so that while removing the elements in the list we don't skip any.
                del datasets_paths[index] #delete the name of the csv file at the current index in indices.
                del features_paths[index] #delete the name of the npz features file at the curre index in indices.

        # load features
        all_features = [] #empty list that will contain the loaded features.
        for _, feature_file_name in enumerate((t := tqdm(features_paths))): #iterate over npz features file names.
            t.set_description(f"Loading features: {feature_file_name}") #update progress bar
            with np.load(feature_file_name) as features: 
                all_features.append(features["arr_0"]) #load the feature file corresponding to current file name.

        all_features = np.concatenate(all_features, axis=0) #concatenate all the loaded features in a single numpy array.

        # load datasets
        dfs = [] #empty list that will contain the dataframes loaded from the selected csv fles.
        for _, filename in enumerate(tqdm(datasets_paths, desc="Loading csv datasets")): #iterate over csv file names.
            df = pd.read_csv(filename, index_col=0) #read csv into dataframe
            dfs.append(df) #append curr dataframe to all dataframes list.
        dataset = pd.concat(dfs, ignore_index=True) #concat all dataframes in the list into one single dataframe.

        # drop unwanted cameras
        names = dataset["frame_name"] # extract frame names from dataset.
        cams = [] #empty list that will store the cam number of each frame name.
        for name in names: 
            # find index of cam namber by finding substr "cam" in frame name and shifting index to + 4.
            index = name.find("cam") + 4 
            cams.append(int(name[index])) # append cam number casted to int.

        dataset["cam"] = pd.Series(cams) #append pd.Series of cam numbers to dataset.

        # drop unwanted cams.
        if self.cams: 
            self.cams = [int(c) for c in self.cams] #cast to int the user submitted cam numbers.
            cams_to_drop_mask = ~dataset["cam"].isin(self.cams) # create a mask with false where the column "cam" of the dataset contains an unwanted cam number.
            dataset = dataset.loc[~cams_to_drop_mask, :] #drop the unwanted rows from dataset.
            dataset.reset_index(drop=True, inplace=True) # reset dataframe index.

            all_features = np.delete(all_features, cams_to_drop_mask.tolist(), axis=0) #remove unwanted features using the same unwanted cams mask.
            all_features = normalize(all_features, axis=1, norm="l1") #normalize features using L1 norm.

        # drop off air frames
        if self.drop_offair:
            offair_mask = dataset["ar_labels"] == "actor_repositioning" #create mask with False where ar label is "actor_repositioning".

            dataset = dataset.loc[~offair_mask, :] #drop "actor_repositioning" frames from dataframe using the created mask.
            dataset.reset_index(drop=True, inplace=True) #reset dataframe index again.
            all_features = np.delete(all_features, offair_mask.tolist(), axis=0) # delete actual features from all features vector using created mask.

        return dataset, all_features # return dataframe and features vector.


def load_and_split(features_folder: str, dataset_folder: str, train_actors: list, val_actors: list, train_cams: list,
                   val_cams: list, split_ratio: float, drop_offair: bool, undersample: bool, micro_classes: bool
                   ) -> tuple[np.ndarray, list, np.ndarray, list, list, list]:
    
    
    # Load dataset and features
    
    if val_actors: #if user inputs some validation actors use them as validation set.
        train_dataloader = DatasetLoader(dataset_folder, features_folder, train_actors, train_cams, drop_offair)# istantiate DataLoader for train set.
        val_dataloader = DatasetLoader(dataset_folder, features_folder, val_actors, val_cams, drop_offair) #istantiate DataLoader for val set.
        print("[STATUS] Load Train Set")
        train_dataset, train_features = train_dataloader.load() # load the train set.
        print("[STATUS] Load Val Set")
        val_dataset, val_features = val_dataloader.load()  # load the val set.

        X_train = train_features
        X_val = val_features
        
        # Use micro or macro labels depending on the user input.
        if micro_classes:
            y_train = train_dataset["micro_labels"].tolist()
            y_val = val_dataset["micro_labels"].tolist()
        else:
            y_train = train_dataset["macro_labels"].tolist()
            y_val = val_dataset["macro_labels"].tolist()

        cams_train = train_dataset["cam"].tolist() # extract cam numbers from train set.
        cams_val = val_dataset["cam"].tolist() # extract cam numbers from val set.
        
        # use NearMiss undersampling algorithm if specified in user input.
        if undersample:
            print("[STATUS] Undersampling train set")
            print(f"Initial Train set distribution: {Counter(y_train)}")
            us = NearMiss(version=1)
            X_train, y_train = us.fit_resample(X_train, y_train)
            print(f"Train set distribution after undersampling: {Counter(y_train)}")

        return normalize(X_train), y_train, normalize(X_val), y_val, cams_train, cams_val # return normalized features, labels, and cam numbers.

    else:
        # do the train-validation split
        dataset_dataloader = DatasetLoader(dataset_folder, features_folder, train_actors, train_cams, drop_offair) # istantiate DataLoader.
        print("[STATUS] Load Dataset")
        
        dataset, features = dataset_dataloader.load() # actually load dataset.
        split = int(dataset.shape[0] * split_ratio)  # define split index based on split_ratio argument.
        
        print("[STATUS] Splitting in Train and Val sets")
        X_train = np.array(features[0:split, :]) #split the train set.
        X_val = np.array(features[split:, :]) # split the val set.

        # load micro or macro labels depending on the user input.
        if micro_classes:
            y_train = dataset["micro_labels"][0:split].tolist()
            y_val = dataset["micro_labels"][split:].tolist()
        else:
            y_train = dataset["macro_labels"][0:split].tolist()
            y_val = dataset["macro_labels"][split:].tolist()

        cams_train = dataset["cams"][0:split].tolist()
        cams_val = dataset["cams"][split:].tolist()
        
        # use NearMiss undersampling algorithm if specified in user input.
        if undersample:
            print("[STATUS] Undersampling train set")
            print(f"Initial Train set distribution: {Counter(y_train)}")
            us = NearMiss(version=1)
            X_train, y_train = us.fit_resample(X_train, y_train)
            print(f"Train set distribution after undersampling: {Counter(y_train)}")

        return normalize(X_train), y_train, normalize(X_val), y_val, cams_train, cams_val # return normalized features, labels, and cam numbers.



def get_timeseries_labels_encoded(y_train, y_val, cfg) -> tuple[list, list, LabelEncoder, list]:
    """
    
    """
    
    def to_series_labels(timestep_labels: list, n_batches: int, n_windows: int, seq_len: int, stride: int) -> list:
        series_labels = []
        s = 0
        for w in range(n_windows * n_batches):
            s = w * stride
            labels_seq = timestep_labels[s: s + seq_len]
            series_labels.append(mode(labels_seq))
        return series_labels

    n_train_batches = len(y_train) // cfg.batch_size
    n_val_batches = len(y_val) // cfg.batch_size
    n_windows = (cfg.batch_size - cfg.seq_len) // cfg.stride + 1

    y_train_series = to_series_labels(y_train, n_train_batches, n_windows, cfg.seq_len, cfg.stride)
    y_val_series = to_series_labels(y_val, n_val_batches, n_windows, cfg.seq_len, cfg.stride)

    print(f"\nBefore ENCODING -- len(y_train_series): {len(y_train_series)} len(y_val_series): {len(y_val_series)}")

    print(f"y_train_series value counts:\n {pd.Series(y_train_series).value_counts()}")
    # encoding
    enc = LabelEncoder()
    enc = enc.fit(y_train_series)
    mapping = dict(zip(enc.classes_, range(1, len(enc.classes_) + 1)))
    print(f"Classes mapping: {mapping}")

    # y_train_series_unique = np.unique(y_train_series)
    # y_val_series_unique = np.unique(y_val_series)

    # if y_train_series_unique.sort() != y_val_series_unique.sort():
    #     raise Exception("y_train_series_unique != y_val_series_unique")

    y_train_series_encoded = enc.fit_transform(y_train_series)
    #class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_series_encoded), y=y_train_series_encoded)
    #d_class_weights = dict(enumerate(class_weights))
    print(f"\nClass weights for train series: {class_weights}")

    y_train_series = enc.fit_transform(y_train_series)
    y_val_series = enc.fit_transform(y_val_series)

    print(f"\nAfter ENCODING -- len(y_train_series): {len(y_train_series)} len(y_val_series): {len(y_val_series)}")
    return y_train_series, y_val_series, enc, enc.classes_.tolist()
