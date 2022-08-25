from statistics import mode

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import trange


class TimeSeriesGenerator:
    """
    Generate time-series data from single frames features.
    -----
    Parameters:
    - opt: dict. Dictionary containing all the arguments entered by the user.
    
    """

    def __init__(self, opt):
        self.labels_encoder = LabelEncoder()
        self.n_windows: int = (opt.batch_size - opt.seq_len) // opt.stride + 1
        self.seq_len = opt.seq_len
        self.batch_size = opt.batch_size
        self.stride = opt.stride
        self.num_features = opt.num_features

    def __to_time_series(self, X: np.array, y: list, cams: list, n_series: int) -> tuple[list, list]:
        """
        Splits features and labels in time-series.
        ----
        Parameters:
        - X: np.array: contains the single frames' features to split in time series.
        - y: list: contains the label of each frame.
        - cams: list: contains the cam number of each frame.
        - n_series: number of time-series (windows) to split the dataset into.        
        """

        # instantiate emtpy list of placeholder arrays with dimension num_features.
        # This list will hold actual features windows, indeed has length equal to the total number of windows obtainable from the dataset.
        time_series: list = [np.empty(self.num_features)] * n_series

        y_s: list = [None] * n_series  # instantiate empty list with length number of windows.
        s: int = 0  # index for splitting in time-series

        for w in trange(n_series):  # iterate over the number of possible windows.
            s = w * self.stride  # update index.
            features_seq = X[s: s + self.seq_len, :]  # select a window of single frames' features.
            labels_seq = y[s: s + self.seq_len]  # select a window of single frames' labels.
            cams_seq = cams[s: s + self.seq_len]  # select a window of single frames' cams.
            curr_cam = mode(cams_seq)  # calculate the most frequent cam number in the window.

            for i, _ in enumerate(cams_seq):  # for each cam number in the window.
                if cams_seq[
                    i] != curr_cam:  # if the cam number is not equal to the most frequent cam number in the window.
                    features_seq[i] = np.zeros(
                        self.num_features)  # pad features with a zeros array with length num_features.
                labels_seq[i] = -10  # insert a negative value at the current label to later remove this unwanted label.

        time_series[w] = features_seq  # update window with padding features.

        # convert frames' labels in one label per window
        labels_seq = [l for l in labels_seq if
                      l != -10]  # remove all labels of frames with cam number different from the most frequent one.
        label = mode(labels_seq)  # calculate the most frequent label in the window.
        y_s[w] = label  # use that most frequent label as label for the entire window.

        return time_series, y_s  # return windows of features and respective windows' labels.


def get_train_series(self, X: np.array, y: list, cams: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates train windows.
    -----
    Parameters:
    - X: np.array: contains the single frames' features.
    - y: list: contains the single frames' labels.
    - cams: list: contains the single frames' labels.
    
    outputs:
    - X_series: np.array: windows of features.
    - y_series: list: windows' labels.
    
    """
    n_batches: int = len(
        y) // self.batch_size  # calculate number of possible batches in the dataset given the batch size.
    n_series: int = self.n_windows * n_batches  # calculate number of possibile windows in the dataset given the number of batches.

    X_series, y_series = self.__to_time_series(X, y, cams, n_series)  # split in windows.

    self.labels_encoder = self.labels_encoder.fit(y_series)  # fit encoder on windows' labels.
    self.classes_ = self.labels_encoder.classes_.tolist()  # add encoded labels as object's property.
    mapping = dict(zip(self.labels_encoder.classes_, range(1,
                                                           len(self.labels_encoder.classes_) + 1)))  # class mappings from categorical to numerical.
    self.mapping = mapping  # add class mappings as object's property.
    y_series = self.labels_encoder.transform(y_series)  # encode windows' labels.

    return np.array(X_series), y_series,  # return windows' features and windows' labels.


def get_val_series(self, X: np.ndarray, y: list, cams: list):
    """
    Generates validation windows.
    -----
    Parameters:
    - X: np.array: contains the single frames' features.
    - y: list: contains the single frames' labels.
    - cams: list: contains the single frames' labels.
    
    Outputs:
    - X_series: np.array: windows of features.
    - y_series: list: windows' labels.
    
    """
    n_batches: int = len(y) // self.batch_size  # calculate number of possible batches in the dataset.
    n_series: int = self.n_windows * n_batches  # calculate number of possible windows in the dataset.

    X_series, y_series = self.__to_time_series(X, y, cams,
                                               n_series)  # split features in windows and calculate windows' labels.
    y_series = self.labels_encoder.fit_transform(y_series)  # encode labels to numerical.

    return np.array(X_series), y_series  # return windows' features and window's labels.
