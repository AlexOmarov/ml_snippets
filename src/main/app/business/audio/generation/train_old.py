import re

import librosa
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dot, Activation
from keras.layers import Input, Conv1D, Dense, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adam

from business.audio.generation.dto.training_dataset import TrainingDataset
from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_setting import TrainingSetting
from business.audio.generation.dto.training_unit import TrainingUnit
from src.main.resource.config import Config


def train(setting: TrainingSetting) -> str:
    normalized_dataset = _get_dataset(_get_training_units(setting))

    model = _get_model(setting.hyper_params_info)

    # Обучение модели
    model.fit(
        x=[normalized_dataset.training_data, normalized_dataset.training_responses],
        y=[normalized_dataset.test_data, normalized_dataset.test_responses],
        batch_size=setting.hyper_params_info.batch_size,
        epochs=setting.hyper_params_info.num_epochs,
        validation_split=0.2,
        callbacks=[ModelCheckpoint(filepath=setting.paths_info.checkpoint_path_template)]
    )

    return model.save(setting.paths_info.model_dir_path)


def _get_training_units(setting: TrainingSetting) -> [TrainingUnit]:
    return []


def _get_dataset(units: [TrainingUnit]) -> TrainingDataset:
    dataset = _form_dataset(units)
    normalized_dataset = _normalize_dataset(dataset)
    return normalized_dataset


def _form_dataset(units: [TrainingUnit]) -> TrainingDataset:
    return TrainingDataset()


def _normalize_dataset(dataset: TrainingDataset) -> TrainingDataset:
    return dataset


def _get_model(hyper_params: TrainingHyperParamsInfo) -> tf.keras.models.Model:
    # Входные данные
    inputs = Input(shape=(tensor_length, max_seq_length), name='inputs')
    text_inputs = Input(shape=(tensor_length, max_seq_length), name='text_inputs')
    # Encoder текста
    encoder = Bidirectional(LSTM(units=tensor_length, return_sequences=True))(text_inputs)
    for _ in range(encoder_layers - 1):
        encoder = Bidirectional(LSTM(units=tensor_length, return_sequences=True))(encoder)
    # Слой внимания
    attention_rnn = LSTM(units=max_seq_length, return_sequences=True, name='attention_rnn')(encoder)
    attention = Dense(units=max_seq_length, activation='tanh', name='attention')(attention_rnn)
    attention = Dot(axes=[2, 2], name='attention_dot')([attention, inputs])
    attention = Activation('softmax', name='attention_softmax')(attention)
    context = Dot(axes=[1, 1], name='context_dot')([attention, encoder])
    # Декодер
    decoder = LSTM(units=max_seq_length, return_sequences=True)(context)
    for _ in range(decoder_layers - 1):
        decoder = LSTM(units=max_seq_length, return_sequences=True)(decoder)
    decoder_output = Dense(units=max_seq_length)(decoder)
    # Пост-обработка
    postnet = Conv1D(filters=max_seq_length, kernel_size=post_kernel_size, padding='same')(decoder)
    postnet = Activation('tanh')(postnet)
    postnet = Conv1D(filters=max_seq_length, kernel_size=post_kernel_size, padding='same')(postnet)
    postnet_output = Activation('tanh')(postnet)

    model = Model(inputs=[inputs, text_inputs], outputs=[decoder_output, postnet_output])
    loss_fun = hyper_params.loss_fun  # MeanSquaredError()
    model.compile(optimizer=Adam(learning_rate=hyper_params.learning_rate), loss=[loss_fun, loss_fun])
    model.summary()
    return model


def _get_datasets(words_file: str, phonemes_file: str, audio_files_dir: str, tensor_length: int) -> tuple:
    processed_audio, transcripts, processed_text = _get_processed_data(words_file, audio_files_dir, tensor_length)

    max_seq_length = _get_vector_max_length(processed_audio, processed_text)

    normalized_audio = _normalize_audio(processed_audio, max_seq_length)
    normalized_text = _normalize_text(phonemes_file, processed_text, max_seq_length, tensor_length)
    audio_array = np.zeros((len(normalized_audio), tensor_length, max_seq_length))
    text_array = np.zeros((len(normalized_text), tensor_length, max_seq_length))

    for k in range(len(normalized_audio)):
        for i in range(tensor_length):
            for j in range(max_seq_length):
                audio_array[k][i][j] = normalized_audio[k][i][j]

    for k in range(len(normalized_text)):
        for i in range(tensor_length):
            for j in range(max_seq_length):
                text_array[k][i][j] = normalized_text[k][i][j]

    return audio_array, text_array, max_seq_length


def _normalize_text(phonemes_file_path, phoneme_sentences, max_seq_length, tensor_length):
    phonemes = list(set("".join(phoneme_sentences)))
    phonemes.sort()
    with open(phonemes_file_path, 'w', encoding='utf-8') as f:
        f.write(f'{phonemes}')
    return [_get_tensor_for_phoneme_sentence(phonemes, phoneme_sentence, max_seq_length, tensor_length) for
            phoneme_sentence in phoneme_sentences]


def _normalize_audio(processed_audio, max_seq_length):
    result = []
    for audio in processed_audio:
        pad_width = max_seq_length - audio.shape[1]
        result.append(np.pad(audio, ((0, 0), (0, pad_width)), mode='constant'))
    return result


def _get_vector_max_length(processed_audio, processed_text) -> int:
    result = 1
    for audio in processed_audio:
        if audio.shape[1] > result:
            result = audio.shape[1]
    for text in processed_text:
        split = re.findall(words_regex, text)
        if len(split) > result:
            result = len(split)
    return result


def _get_processed_data(words_file, audio_files_dir, tensor_length):
    with open(words_file, 'r', encoding='utf-8') as f:
        data = [line.strip().split('|') for line in f]

    audio_files, transcripts, processed_text = zip(*data)
    audio_files = [audio_files_dir + s for s in audio_files]
    processed_audio = [_form_mel_spec_db(f, tensor_length) for f in audio_files]
    return processed_audio, transcripts, processed_text


def _form_mel_spec_db(audio_path, num_mels, sampling_rate=Config.AUDIO_GENERATION_SAMPLE_RATE, duration=4):
    # Load the audio file
    audio_data, sr = librosa.load(audio_path, sr=sampling_rate, duration=duration, mono=True)

    # Normalize the audio
    audio_data /= max(abs(audio_data))

    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Resample the audio if necessary
    if sr != sampling_rate:
        audio_data = librosa.resample(audio_data, sr, sampling_rate)

    # Apply the Mel spectrogram transformation
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=num_mels, fmin=125, fmax=7600)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def _get_tensor_for_phoneme_sentence(one_hot, processed_text, desired_length, tensor_length):
    words = re.findall(words_regex, processed_text)
    result = []
    for phoneme in one_hot:
        vector = []
        for word in words:
            if phoneme in word:
                contains = 1
            else:
                contains = 0
            vector.append(contains)
        while len(vector) < desired_length:
            vector.append(0)
        result.append(vector)
    while len(result) < tensor_length:
        result.append([0] * desired_length)
    return result
