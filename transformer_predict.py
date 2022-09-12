import json
from glob import glob

import numpy as np

from audio_processing import load_audio_file, random_crop
from models import transformer_decoder

transformer_h5 = "transformer_decoder.h5"

CLASS_MAPPING = json.load(open("data/fma_metadata/mapping.json"))

files = sorted(list(glob("data/test/*.*")))

data = [load_audio_file(x, input_length=16000 * 120) for x in files]

transformer_model = transformer_decoder(n_classes=len(CLASS_MAPPING))

transformer_model.load_weights(transformer_h5)

crop_size = np.random.randint(128, 512)
repeats = 10

transformer_v2_Y = 0

for _ in range(repeats):
    X = np.array([random_crop(x, crop_size=crop_size) for x in data])

    transformer_v2_Y += transformer_model.predict(X) / repeats

transformer_v2_Y = transformer_v2_Y.tolist()

for path, pred in zip(files, transformer_v2_Y):
    print(path)
    pred_tup = [(k, pred[v]) for k, v in CLASS_MAPPING.items()]
    pred_tup.sort(key=lambda x: x[1], reverse=True)

    for a in pred_tup[:5]:
        print(a)
    print()
