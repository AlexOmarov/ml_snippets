import os
import re

import librosa
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Concatenate, Softmax, Dot
from keras.layers import Input, Conv1D, Dense, Dropout, LSTM, Bidirectional
from keras.models import Model

from src.main.resource.config import Config


def train(words_file: str = Config.WORDS_FILE_PATH,
          audio_files_dir: str = Config.AUDIO_FILES_DIR_PATH,
          phonemes_file: str = Config.PHONEMES_FILE_PATH,
          learning_rate: str = Config.AUDIO_GENERATION_LEARNING_RATE,
          checkpoint_dir_path: str = Config.AUDIO_GENERATION_CHECKPOINT_DIR_PATH,
          num_epochs: str = Config.AUDIO_GENERATION_NUM_EPOCHS,
          num_mels: int = Config.AUDIO_GENERATION_NUM_MELS,
          vocab_size: int = Config.AUDIO_GENERATION_VOCAB_SIZE):
    tensor_length = max(vocab_size, num_mels)

    audio_dataset, text_dataset = _get_datasets(words_file, phonemes_file, audio_files_dir, tensor_length)

    dataset = tf.data.Dataset.zip((audio_dataset, text_dataset)).map(
        lambda audio, text: ({'input_1': audio, 'input_2': text}))

    model = _get_model(tensor_length)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    checkpoint_name = 'model.{epoch:02d}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir_path, checkpoint_name))

    # dataset = dataset.batch(batch_size)
    model.fit(dataset, epochs=num_epochs, callbacks=[checkpoint])


def generate():
    print()


def _get_model(tensor_length) -> tf.keras.models.Model:
    # Encoder
    encoder_input = Input(shape=(None, 80))  # shape: (batch_size, time_steps, num_mels)
    encoder_conv1 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(encoder_input)
    encoder_conv2 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(encoder_conv1)
    encoder_conv3 = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(encoder_conv2)

    encoder_lstm1 = Bidirectional(LSTM(units=256, return_sequences=True))(encoder_conv3)
    encoder_lstm2 = Bidirectional(LSTM(units=256, return_sequences=True))(encoder_lstm1)

    # Decoder
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=tensor_length, output_dim=256)(decoder_input)

    decoder_lstm1 = LSTM(units=1024, return_sequences=True)(decoder_embedding)
    decoder_lstm2 = LSTM(units=1024, return_sequences=True)(decoder_lstm1)

    # Attention
    attention_hidden = Dense(units=512, activation='tanh')(encoder_lstm2)
    attention_scores = Dot(axes=[2, 1])([attention_hidden, decoder_lstm2])
    attention_weights = Softmax()(attention_scores)
    context_vector = Dot(axes=[2, 1])([encoder_lstm2, attention_weights])
    attention_combined = Concatenate(axis=-1)([decoder_lstm2, context_vector])

    attention_dense = Dense(units=256, activation='tanh')(attention_combined)
    attention_dropout = Dropout(rate=0.1)(attention_dense)
    decoder_output = Dense(units=tensor_length, activation='softmax')(attention_dropout)

    # Model
    model = Model([encoder_input, decoder_input], decoder_output)

    return model


def _get_datasets(words_file: str, phonemes_file: str, audio_files_dir: str, tensor_length: int) -> tuple:
    processed_audio, transcripts, processed_text = _get_processed_data(words_file, audio_files_dir, tensor_length)

    max_seq_length = _get_vector_max_length(processed_audio, processed_text)

    normalized_audio = _normalize_audio(processed_audio, max_seq_length)
    normalized_text = _normalize_text(phonemes_file, processed_text, max_seq_length, tensor_length)

    audio_dataset = tf.data.Dataset.from_tensor_slices(normalized_audio)
    text_dataset = tf.data.Dataset.from_tensor_slices(normalized_text)

    return audio_dataset, text_dataset


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
        split = re.findall(r'\b\w+\b', text)
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


def _form_mel_spec_db(audio_path, num_mels, sampling_rate=22050, duration=4):
    # Load the audio file
    print("Got preprocess_audio for " + audio_path)
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
    words = re.findall(r'\b\w+\b', processed_text)
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


train()
