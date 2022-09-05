from util.ml_logger import logger

# parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__'].__name__.replace('__', '\'')
logger = logger.get_logger(__name__.replace('__', '\''))  # Create logger for current running script
