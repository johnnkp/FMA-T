# https://github.com/CVxTz/music_genre_classification/blob/master/code/trsf_pretrain.py
import json
from glob import glob
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from models import transformer_pretrain
from prepare_data import get_id_from_path, PretrainGenerator

if __name__ == "__main__":
    from collections import Counter

    h5_name = "transformer_pretrain.h5"
    batch_size = 32
    epochs = 20
    id_to_genres = json.load(open("data/fma_metadata/tracks_genre.json"))
    id_to_genres = {int(k): v for k, v in id_to_genres.items()}

    base_path = os.path.join(os.getcwd(), 'data', os.listdir('data')[1], 'fma_large')
    files = sorted(list(glob(base_path + "/*/*.npy")))
    files = [x for x in files if id_to_genres[int(get_id_from_path(x))]]
    labels = [id_to_genres[int(get_id_from_path(x))] for x in files]
    print(len(labels))

    samples = list(zip(files, labels))

    strat = [a[-1] for a in labels]
    cnt = Counter(strat)
    strat = [a if cnt[a] > 2 else "" for a in strat]

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337, stratify=strat
    )

    model = transformer_pretrain()

    try:
        model.load_weights(h5_name, by_name=True)
        print("Weights Loaded")
    except:
        print("Could not load weights")

    checkpoint = ModelCheckpoint(
        h5_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    reduce_o_p = ReduceLROnPlateau(
        monitor="val_loss", patience=20, min_lr=1e-7, mode="min"
    )

    try:
        model.load_weights(h5_name)
    except:
        print("Could not load weights")

    model.fit_generator(
        PretrainGenerator(train, batch_size=batch_size),
        validation_data=PretrainGenerator(val, batch_size=batch_size),
        epochs=epochs,
        callbacks=[checkpoint, reduce_o_p],
        max_queue_size=64,
    )
