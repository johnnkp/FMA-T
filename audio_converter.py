# https://github.com/CVxTz/music_genre_classification/blob/master/code/audio_processing.py
import numpy as np
from matplotlib import pyplot as plt

from audio_processing import load_audio_file, random_mask, random_crop


def convert(path):
    data = load_audio_file(path)

    print(data.shape)

    new_data = random_mask(data)

    plt.figure()
    plt.imshow(data.T)
    plt.show()
    plt.savefig(path + ".png")

    plt.figure()
    plt.imshow(new_data.T)
    plt.show()
    plt.savefig(path + "_random_mask.png")

    print(np.min(data), np.max(data))

    np.save(path.replace(".mp3", ".npy"), data)

    crop_size = np.random.randint(128, 256)
    crop = random_crop(np.load(path.replace(".mp3", ".npy")), crop_size=crop_size)

    plt.figure()
    plt.imshow(crop.T)
    plt.show()
    plt.savefig(path + "_random_crop.png")
