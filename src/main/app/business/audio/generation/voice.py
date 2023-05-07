import ast
import os
import re

import librosa
import numpy as np
import pymorphy2
import tensorflow as tf
from keras.layers import Dot, Activation
from keras.layers import Input, Conv1D, Dense, LSTM, Bidirectional
from keras.losses import MeanSquaredError
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from phonemizer import phonemize

from src.main.resource.config import Config

words_regex = r'\b\w+\b'


def train(words_file: str = Config.WORDS_FILE_PATH,
          audio_files_dir: str = Config.AUDIO_FILES_DIR_PATH,
          batch_size: str = Config.AUDIO_GENERATION_BATCH_SIZE,
          phonemes_file: str = Config.PHONEMES_FILE_PATH,
          learning_rate: str = Config.AUDIO_GENERATION_LEARNING_RATE,
          checkpoint_dir_path: str = Config.AUDIO_GENERATION_CHECKPOINT_DIR_PATH,
          num_epochs: str = Config.AUDIO_GENERATION_NUM_EPOCHS,
          num_mels: int = Config.AUDIO_GENERATION_NUM_MELS,
          model_path: str = Config.AUDIO_MODEL_PATH,
          vocab_size: int = Config.AUDIO_GENERATION_VOCAB_SIZE):
    tensor_length = max(vocab_size, num_mels)

    audio_dataset, text_dataset, max_seq_length = _get_datasets(words_file, phonemes_file, audio_files_dir,
                                                                tensor_length)
    encoder_layers = 3
    decoder_layers = 2
    post_kernel_size = 5

    # Создание модели
    model = _get_model(tensor_length, encoder_layers, max_seq_length, decoder_layers, post_kernel_size)

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=[MeanSquaredError(), MeanSquaredError()])

    checkpoint_name = 'model.{epoch:02d}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir_path, checkpoint_name))

    # Обучение модели
    model.fit(x=[text_dataset, audio_dataset], y=[audio_dataset, audio_dataset], batch_size=batch_size,
              epochs=num_epochs,
              validation_split=0.1, callbacks=[checkpoint])

    model.save(model_path)


def generate(text: str, model_path: str = Config.AUDIO_MODEL_PATH,
             num_mels: int = Config.AUDIO_GENERATION_NUM_MELS,
             phonemize_language: str = Config.PHONEMIZE_LANGUAGE,
             phonemize_backend=Config.PHONEMIZE_BACKEND,
             phonemes_file: str = Config.PHONEMES_FILE_PATH,
             vocab_size: int = Config.AUDIO_GENERATION_VOCAB_SIZE):
    model = load_model(model_path)
    input_shape = model.layers[0].input_shape
    desired_length = input_shape[-1][-1]
    morph = pymorphy2.MorphAnalyzer()
    tensor_length = max(vocab_size, num_mels)
    with open(phonemes_file, 'r', encoding='utf-8') as f:
        phonemes = [ast.literal_eval(line) for line in f][0]

    words = re.findall(words_regex, text)
    all_phonemes = []

    for word in words:
        base_form = morph.parse(word)[0].normal_form
        phonemes = phonemize(base_form, language=phonemize_language, backend=phonemize_backend)
        all_phonemes.append(phonemes)

    processed = ''.join(all_phonemes)

    input_for_model = _get_tensor_for_phoneme_sentence(phonemes, processed, desired_length, tensor_length)

    # Generate speech using model and input_for_model
    output_tensor = model.predict(input_for_model)

    # Post-process the output tensor
    mel_spec = output_tensor[0]
    linear_spec = output_tensor[1]
    mel_spec = np.squeeze(mel_spec, axis=0)
    linear_spec = np.squeeze(linear_spec, axis=0)
    mel_spec = librosa.db_to_amplitude(mel_spec)
    librosa.db_to_amplitude(linear_spec)

    # Generate speech from the post-processed tensor
    waveform = librosa.feature.inverse.mel_to_audio(mel_spec, sr=Config.AUDIO_GENERATION_SAMPLE_RATE)
    return waveform


def _get_model(tensor_length, encoder_layers, max_seq_length, decoder_layers,
               post_kernel_size) -> tf.keras.models.Model:
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
    # Модель
    model = Model(inputs=[inputs, text_inputs], outputs=[decoder_output, postnet_output])
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


generate("Привет!")
