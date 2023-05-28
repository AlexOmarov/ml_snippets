from business.audio.generation.train.config.training_setting import ts
from business.audio.generation.train.dataset.dataset_former import form_dataset
from business.audio.generation.train.model.model_creator import create_model

form_dataset(ts)
create_model(ts)
