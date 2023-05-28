from business.audio.generation.config.dto.training_hyper_params_info import TrainingHyperParamsInfo
from business.audio.generation.config.dto.training_paths_info import TrainingPathsInfo
from business.audio.generation.config.dto.training_setting import TrainingSetting
from src.main.resource.config import Config

ts = TrainingSetting(
    hyper_params_info=TrainingHyperParamsInfo(
        batch_size=Config.AUDIO_GENERATION_BATCH_SIZE,
        num_epochs=Config.AUDIO_GENERATION_NUM_EPOCHS,
        steps_per_epoch=Config.AUDIO_GENERATION_STEPS_PER_EPOCH
    ),
    paths_info=TrainingPathsInfo(
        metadata_file_path=Config.METADATA_FILE_PATH,
    ),
    model_name=Config.AUDIO_GENERATION_MODEL_NAME,
    language=Config.PHONEMIZE_LANGUAGE,
)
