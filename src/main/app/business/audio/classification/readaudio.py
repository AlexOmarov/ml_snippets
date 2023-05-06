"""
Script for generating sound file from image of sinusoid signal

This script allows user to classify music into the corresponding genre.

This tool accepts image file as a parameter.

This file can also be imported as a module.

Public interface:
    * read - gets an image with sinusoid and converts it to audio file
"""

from werkzeug.datastructures import FileStorage

from business.util.ml_logger import logger
from presentation.api.audio_generation_result import AudioGenerationResult

_log = logger.get_logger(__name__.replace('__', '\''))


def read(image: FileStorage) -> AudioGenerationResult:
    """
    Makes a prediction based on passed image.

    :param image: Storage with image data
    """

    return AudioGenerationResult(data=image.filename)
