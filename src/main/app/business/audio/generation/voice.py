import os

import librosa
import numpy as np
import phonemizer
import tensorflow as tf

from src.main.resource.config import Config
from keras.preprocessing.text import Tokenizer

# Define the hyperparameters
learning_rate = 0.001
batch_size = 16
num_epochs = 50
num_mels = 80  # number of mel spectrogram bins
vocab_size = 50
checkpoint_dir = 'checkpoints'

root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
input_wav_dir = os.path.join(root_dir, Config.VOICE_PATH)
input_descr_dir = os.path.join(root_dir, 'data')

# Load the data
with open(os.path.join(input_descr_dir, 'words.txt'), 'r', encoding='utf-8') as f:
    data = [line.strip().split('|') for line in f]
audio_files, transcripts = zip(*data)
audio_files = [input_wav_dir + s for s in audio_files]


# Define the preprocessing functions
def preprocess_audio(audio_path, sampling_rate=22050, duration=4):
    # Load the audio file
    print("Got preprocess_audio for " + audio_path)
    audio, sr = librosa.load(audio_path, sr=sampling_rate, duration=duration, mono=True)

    # Normalize the audio
    audio /= max(abs(audio))

    # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)

    # Resample the audio if necessary
    if sr != sampling_rate:
        audio = librosa.resample(audio, sr, sampling_rate)

    # Apply the Mel spectrogram transformation
    mel_spec = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_mels=num_mels, fmin=125, fmax=7600)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def preprocess_text(text):
    # Define the phonemizer language model for Russian language
    language_model = phonemizer.LanguageModel(
        phonemizer.phonemize.Language.RUSSIAN,
        phonemizer.phonemize.Language.RUSSIAN
    )

    # Tokenize the text into individual words
    words = text.strip().split()

    # Convert each word to its corresponding sequence of phonemes
    phonemes = []
    for word in words:
        word_phonemes = phonemizer.phonemize(word, language_model=language_model)
        phonemes.append(word_phonemes)

    # Combine the phonemes into a single sequence
    text_phonemes = ' '.join(phonemes)

    return text_phonemes


def transcript_generator():
    for transcript in transcripts:
        yield preprocess_text(transcript)


# Create the dataset
processed_audio = [preprocess_audio(f) for f in audio_files]
# Pad the audio files to the same length
max_seq_length = 500
padded_audio = []
for audio in processed_audio:
    pad_width = max_seq_length - audio.shape[1]
    padded_audio.append(np.pad(audio, ((0, 0), (0, pad_width)), mode='constant'))

audio_dataset = tf.data.Dataset.from_tensor_slices(padded_audio)

text_dataset = tf.data.Dataset.from_generator(transcript_generator, output_types=tf.string, output_shapes=None)

dataset = tf.data.Dataset.zip((audio_dataset, text_dataset)).map(
    lambda audio, text: ({'input_1': audio, 'input_2': text}))
dataset = dataset.batch(batch_size)

# Build the model

inputs = tf.keras.layers.Input(shape=(None,))
embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256)(inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(1024, return_sequences=True, return_state=True)(embedding)
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(1024, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
outputs = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([inputs, decoder_inputs], outputs)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Train the model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'model.{epoch:02d}.h5'))
model.fit(dataset, epochs=num_epochs, callbacks=[checkpoint_callback])

# Synthesize speech
# TODO: Implement speech synthesis function using the trained model
