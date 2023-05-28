import pickle

import numpy as np
from keras.models import load_model

from business.audio.generation.config.training_setting import ts
from business.audio.generation.dto.audio_entry import AudioEntry
from business.audio.generation.dto.training_setting import TrainingSetting


def identify(setting: TrainingSetting):
    model = load_model(setting.paths_info.model_dir_path + "speaker_verification")
    filename = f"/serialized_batch_{4}.pkl"
    with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
        units: [AudioEntry] = pickle.load(file)
        result = model.predict(units[0].feature_vector[None, :])
        max_index = np.argmax(result)
        print(result)
        print(max_index)
        print(np.where(units[0].speaker_identification_vector == 1)[0])


identify(ts)
