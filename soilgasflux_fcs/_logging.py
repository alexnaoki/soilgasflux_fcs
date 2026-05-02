import logging

# Python std-library guidance for libraries: attach a NullHandler so that log
# records are silently discarded when the application hasn't configured logging.
# We used to auto-install a StreamHandler(stderr), which leaked warnings and
# tracebacks into Jupyter cells. Applications that want console output should
# call `logging.basicConfig(...)` or attach their own handler.
_PKG_LOGGER_NAME = 'soilgasflux_fcs'


def get_logger(name: str) -> logging.Logger:
    pkg_logger = logging.getLogger(_PKG_LOGGER_NAME)
    if not any(isinstance(h, logging.NullHandler) for h in pkg_logger.handlers):
        pkg_logger.addHandler(logging.NullHandler())
    return logging.getLogger(name)
