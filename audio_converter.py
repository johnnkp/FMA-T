# https://github.com/CVxTz/music_genre_classification/blob/master/code/audio_processing.py
import numpy as np
from matplotlib import pyplot as plt

from audio_processing import load_audio_file, random_mask


def convert(path):
    data = load_audio_file(path)

    print(data.shape)

    new_data = random_mask(data)

    plt.figure()
    plt.imshow(data.T)
    plt.show()

    plt.figure()
    plt.imshow(new_data.T)
    plt.show()

    print(np.min(data), np.max(data))

    np.save(path.replace(".mp3", ".npy"), data)
