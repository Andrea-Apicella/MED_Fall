import glob
import os

from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def listdir_nohidden_sorted(path) -> list:
    """Returns a list of the elements in the specified path, sorted by name. Skips dotfiles."""
    l = glob.glob(os.path.join(path, '*'))
    if len(l) > 0:
        return natsorted(l)
    else:
        raise Exception(f'List is empty. Invalid path {path}')


def safe_mkdir(path) -> bool:
    """If does not already exists, makes a directory in the specified path and returns True. Else returns False."""
    if not os.path.isdir(path):
        os.makedirs(path)
        return True
    else:
        return False
    
    
def load_images(images_dir: str, images_names: list, extension = None) -> np.array:
    """Loads images as numpy array. 
    
    Parameters
    ----------
    images_dir: str. Path pointing to the folder that contains the images.
    
    images_names: list containing the titles of the images.
    
    """
    if extension is None:
        images = np.array([imread(f"{images_dir}/{name}") for name in images_names])
    else: 
        images = np.array([imread(f"{images_dir}/{name}.{extension}") for name in images_names])
    print(images.shape)
    return images
    
def show_images(images, rows=3, titles=None, figsize=(15, 10)):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    rows (Default = 1): Number of columns in figure (number of rows is
                        set to int(np.ceil(n_images/float(rows)))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert (titles is None) or (len(images) == len(titles))
    n_images = len(images)
    if titles is None:
        titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, int(n_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis("off")
        a.set_title(title)
    plt.show()


