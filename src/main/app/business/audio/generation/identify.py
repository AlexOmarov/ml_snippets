import pickle

import numpy as np
from keras.models import load_model

from business.audio.generation.train.config.training_setting import ts
from business.audio.generation.train.dataset.dataset_former import form_dataset
from business.audio.generation.train.dataset.dto.audio_entry import AudioEntry
from business.audio.generation.train.model.model_creator import create_model


def identify(model_dir_path: str, serialized_units_dir_path: str, filename: str):
    model = load_model(model_dir_path + "speaker_verification")
    with open(serialized_units_dir_path + filename, 'rb') as file:
        units: [AudioEntry] = pickle.load(file)
    result = model.predict(units[0].feature_vector[None, :])
    max_index = np.argmax(result)
    print("Got: " + max_index.__str__() + ", valid: " + np.where(units[0].speaker_identification_vector == 1)[0])


# form_dataset(ts)
create_model(ts)
