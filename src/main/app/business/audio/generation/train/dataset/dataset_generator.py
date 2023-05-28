import os
import pickle

import numpy as np

from business.audio.generation.train.train import AudioEntry


def get_dataset_generator(serialized_units_dir_path: str, init_batch: list):
    batches_amount = len(os.listdir(serialized_units_dir_path))
    while True:
        if batches_amount <= init_batch[0]:
            init_batch[0] = 4
        filename = f"/serialized_batch_{init_batch[0]}.pkl"
        with open(serialized_units_dir_path + filename, 'rb') as file:
            units: [AudioEntry] = pickle.load(file)
        batch_features = [unit.feature_vector for unit in units]
        batch_identification_vectors = [unit.speaker_identification_vector for unit in units]
        init_batch[0] = init_batch[0] + 1
        yield np.array(batch_features), np.array(batch_identification_vectors)
