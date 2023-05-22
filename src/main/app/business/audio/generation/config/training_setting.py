from business.audio.generation.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.dto.training_paths_info import TrainingPathsInfo
from business.audio.generation.dto.training_setting import TrainingSetting
from src.main.resource.config import Config

ts = TrainingSetting(
    hyper_params_info=TrainingHyperParamsInfo(
        batch_size=Config.AUDIO_GENERATION_BATCH_SIZE,
        learning_rate=Config.AUDIO_GENERATION_LEARNING_RATE,
        num_epochs=Config.AUDIO_GENERATION_NUM_EPOCHS,
        loss_fun=Config.AUDIO_GENERATION_LOSS_FUN,
        validation_split=Config.AUDIO_GENERATION_VALIDATION_SPLIT,
        encoder_layers=Config.AUDIO_GENERATION_ENCODER_LAYERS,
        decoder_layers=Config.AUDIO_GENERATION_DECODER_LAYERS,
        post_kernel_size=Config.AUDIO_GENERATION_POST_KERNEL_SIZE,
        steps_per_epoch=Config.AUDIO_GENERATION_STEPS_PER_EPOCH
    ),
    paths_info=TrainingPathsInfo(
        metadata_file_path=Config.METADATA_FILE_PATH,
        phonemes_file_path=Config.PHONEMES_FILE_PATH,
        audio_files_dir_path=Config.AUDIO_FILES_DIR_PATH,
        checkpoint_path_template=Config.AUDIO_GENERATION_CHECKPOINT_FILE_PATH_TEMPLATE,
        serialized_units_dir_path=Config.AUDIO_GENERATION_PICKLED_UNITS_DIR_PATH,
        speaker_file_path=Config.SPEAKER_FILE_PATH,
        model_dir_path=Config.MODEL_DIR_PATH
    ),
    model_name=Config.AUDIO_GENERATION_MODEL_NAME,
    num_mels=Config.AUDIO_GENERATION_NUM_MELS,
    frame_length=Config.AUDIO_GENERATION_FRAME_LENGTH,
    hop_length=Config.AUDIO_GENERATION_HOP_LENGTH,
    phonemize_language=Config.PHONEMIZE_LANGUAGE,
    vocab_size=Config.AUDIO_GENERATION_VOCAB_SIZE
)
