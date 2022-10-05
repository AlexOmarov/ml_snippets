import logging

import business.util.ml_logger.logger as log_module


class TestLogger:
    def setup(self):
        print("basic setup into class")

    def teardown(self):
        print("basic teardown into class")

    @classmethod
    def setup_class(cls):
        print("class setup")

    @classmethod
    def teardown_class(cls):
        print("class teardown")

    def setup_method(self):
        print("method setup")

    def teardown_method(self):
        print("method teardown")

    def test_created_logger(self):
        logger_name = "TEST_LOGGER"
        log = log_module.get_logger(logger_name)

        assert (log is not None)
        assert (log.name == logger_name)
        assert (log.level == logging.INFO)
