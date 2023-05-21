from business.audio.generation.config.training_setting import ts
from business.audio.generation.speaker_verification.dataset_former import preprocess_audio
from business.audio.generation.speaker_verification.trainer import train

preprocess_audio(ts)
train(ts)


