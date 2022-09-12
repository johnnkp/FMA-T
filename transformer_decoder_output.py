import json
import os
from collections import Counter
from glob import glob

from sklearn.model_selection import train_test_split
import tensorflow as tf
for CUDA in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(CUDA, True)
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from models import transformer_decoder
from prepare_data import get_id_from_path, DataGenerator

h5_name = "transformer_decoder.h5"
h5_pretrain = "transformer_pretrain.h5"
batch_size = 32
epochs = 15
CLASS_MAPPING = json.load(open("data/fma_metadata/mapping.json"))
id_to_genres = json.load(open("data/fma_metadata/tracks_genre.json"))
id_to_genres = {int(k): v for k, v in id_to_genres.items()}

base_path = os.path.join(os.getcwd(), 'data', os.listdir('data')[1], 'fma_large')
files = sorted(list(glob(base_path + "/*/*.npy")))
files = [x for x in files if id_to_genres[int(get_id_from_path(x))]]
labels = [id_to_genres[int(get_id_from_path(x))] for x in files]

samples = list(zip(files, labels))

strat = [a[-1] for a in labels]
cnt = Counter(strat)
strat = [a if cnt[a] > 2 else "" for a in strat]

train, val = train_test_split(samples, test_size=0.2, random_state=1337, stratify=strat)

model = transformer_decoder(n_classes=len(CLASS_MAPPING))

if h5_pretrain:
    model.load_weights(h5_pretrain, by_name=True)
else:
    print("No pretrained encoder.")

checkpoint = ModelCheckpoint(
    h5_name,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

reduce_o_p = ReduceLROnPlateau(
    monitor="val_loss", patience=20, min_lr=1e-7, mode="min"
)

model.fit(
    DataGenerator(train, batch_size=batch_size, class_mapping=CLASS_MAPPING),
    validation_data=DataGenerator(
        val, batch_size=batch_size, class_mapping=CLASS_MAPPING
    ),
    epochs=epochs,
    callbacks=[checkpoint, reduce_o_p],
    max_queue_size=64,
)
