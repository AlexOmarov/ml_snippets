import ast
import re

import librosa
import numpy as np
import pymorphy2
from keras.models import load_model
from phonemizer import phonemize
from scipy.io import wavfile

from src.main.resource.config import Config

words_regex = r'\b\w+\b'


def generate(text: str,
             model_path: str = Config.AUDIO_MODEL_PATH,
             num_mels: int = Config.AUDIO_GENERATION_NUM_MELS,
             words_file: str = Config.METADATA_FILE_PATH,
             audio_files_dir: str = Config.AUDIO_FILES_DIR_PATH,
             phonemize_language: str = Config.PHONEMIZE_LANGUAGE,
             phonemize_backend=Config.PHONEMIZE_BACKEND,
             phonemes_file: str = Config.PHONEMES_FILE_PATH,
             vocab_size: int = Config.AUDIO_GENERATION_VOCAB_SIZE):
    """
    Makes a generation based on passed audio.

    :param vocab_size:
    :param phonemes_file:
    :param phonemize_backend:
    :param phonemize_language:
    :param audio_files_dir:
    :param words_file:
    :param model_path:
    :param text:
    :param num_mels:
    :param audio: Storage with audio data
    """
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

    audio_dataset, text_dataset, max_seq_length = _get_datasets(words_file, phonemes_file, audio_files_dir,
                                                                tensor_length)

    # Here wrong first dimension, then need to add voice file
    input_for_model = _get_tensor_for_phoneme_sentence(phonemes, processed, desired_length, tensor_length)
    text_array = np.zeros((1, tensor_length, max_seq_length))
    for i in range(tensor_length):
        for j in range(max_seq_length):
            text_array[0][i][j] = input_for_model[i][j]

    # Generate speech using model and input_for_model
    output_tensor = model.predict([text_array, np.expand_dims(audio_dataset[0], axis=0)])

    # Post-process the output tensor
    mel_spec = output_tensor[0]
    linear_spec = output_tensor[1]
    mel_spec = np.squeeze(mel_spec, axis=0)
    linear_spec = np.squeeze(linear_spec, axis=0)
    mel_spec = librosa.db_to_amplitude(mel_spec)
    librosa.db_to_amplitude(linear_spec)

    # Generate speech from the post-processed tensor
    waveform = librosa.feature.inverse.mel_to_audio(mel_spec, sr=Config.AUDIO_GENERATION_SAMPLE_RATE)
    waveform = (waveform * 32767).astype('int16')

    wavfile.write(Config.AUDIO_GENERATION_OUTPUT_FILE_PATH, Config.AUDIO_GENERATION_SAMPLE_RATE, waveform)
    return waveform
