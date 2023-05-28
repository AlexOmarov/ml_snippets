from business.audio.generation.train.train import TrainingHyperParamsInfo
from business.audio.generation.train.train import TrainingPathsInfo
from business.audio.generation.train.train import TrainingSetting
from src.main.resource.config import Config

ts = TrainingSetting(
    hyper_params_info=TrainingHyperParamsInfo(
        batch_size=Config.AG_BATCH_SIZE,
        num_epochs=Config.AG_NUM_EPOCHS,
        steps_per_epoch=Config.AG_STEPS_PER_EPOCH,
        num_mels=Config.AG_NUM_MELS,
        frame_length=Config.AG_FRAME_LENGTH,
        hop_length=Config.AG_HOP_LENGTH
    ),
    paths_info=TrainingPathsInfo(
        metadata_file_path=Config.AG_METADATA_FILE_PATH,
        models_dir_path=Config.MODELS_DIR_PATH,
        audio_files_dir_path=Config.AG_AUDIO_DIR_PATH,
        serialized_units_dir_path=Config.AG_SERIALIZED_UNITS_DIR_PATH
    ),
    model_name=Config.AG_MODEL_NAME,
    language=Config.AG_MODEL_LANG,
    words_regex=Config.AG_WORDS_REGEX
)
