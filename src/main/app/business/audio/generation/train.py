import os
import pickle

import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, LSTM, concatenate, TimeDistributed, Masking
from keras.models import Model
from keras.optimizers import Adam

from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.training_unit import TrainingUnit
from business.audio.generation.training_setting import ts
from business.util.ml_logger import logger

_log = logger.get_logger(__name__.replace('__', '\''))


def train(setting: TrainingSetting):
    batches_amount = len(os.listdir(setting.paths_info.serialized_units_dir_path))

    model = _get_model(setting)
    generator = _get_dataset_generator(setting)
    model.fit_generator(
        generator=generator,
        steps_per_epoch=batches_amount,
        epochs=setting.hyper_params_info.num_epochs,
        shuffle=True
    )


# First get dataset, then get model, then train model

def _get_model(setting: TrainingSetting) -> tf.keras.models.Model:
    # Input layer for MFCC db tensor
    mfcc_input = Input(shape=(setting.num_mels, None, 1), name='mfcc_input')
    # Add a masking layer to handle variable-length sequences
    masked_mfcc = Masking()(mfcc_input)

    # Convolutional layers for MFCC feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(masked_mfcc)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    flatten_mfcc = Flatten()(conv2)

    # Reshape flatten_mfcc to match the number of timesteps
    timesteps = None  # Set the appropriate number of timesteps
    reshaped_mfcc = tf.keras.layers.Reshape((timesteps, -1))(flatten_mfcc)

    # Input layer for phonemes
    phoneme_input = Input(shape=(None, 1), name='phoneme_input')
    masked_phoneme = Masking()(phoneme_input)

    # LSTM layer for MFCC processing with variable-length sequences
    lstm_mfcc = LSTM(units=64, return_sequences=True)(reshaped_mfcc)
    # Extract the last output of LSTM for fixed-length output
    mfcc_output = tf.keras.layers.Lambda(lambda x: x[:, :, -1, :])(lstm_mfcc)

    # LSTM layer for phoneme processing
    lstm_phoneme = LSTM(units=64)(masked_phoneme)
    # Extract the last output of LSTM for fixed-length output
    phoneme_output = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(lstm_phoneme)

    # Concatenate MFCC and phoneme outputs
    merged = concatenate([mfcc_output, phoneme_output])

    # TimeDistributed dense layer to apply dense layer to each time step
    dense_time = TimeDistributed(Dense(units=64))(merged)

    # LSTM layer for generating complex spectrogram with variable-length sequences
    lstm_complex_spec = LSTM(units=64, return_sequences=True)(dense_time)

    # Output layer (setting.frame_length // 2 + 1) - amount of frequency bins in result spectrogram
    output = TimeDistributed(Dense(units=setting.frame_length // 2 + 1, activation='linear'))(lstm_complex_spec)
    model = Model(inputs=[mfcc_input, phoneme_input], outputs=output)
    loss_fun = setting.hyper_params_info.loss_fun  # MeanSquaredError()
    model.compile(optimizer=Adam(learning_rate=setting.hyper_params_info.learning_rate), loss=[loss_fun, loss_fun])
    model.summary()
    return model


def _get_dataset_generator(setting: TrainingSetting):
    batch_number = 1
    while True:
        filename = f"/serialized_batch_{batch_number}.pkl"
        with open(setting.paths_info.serialized_units_dir_path + filename, 'rb') as file:
            units: [TrainingUnit] = pickle.load(file)
        batch_mfcc = [unit.mfcc_db for unit in units]
        batch_phonemes = [unit.phonemes for unit in units]
        batch_spectrogram = [unit.spectrogram for unit in units]
        yield [batch_mfcc, batch_phonemes], batch_spectrogram
        batch_number += 1


train(ts)
