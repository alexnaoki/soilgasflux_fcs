import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("soilgasflux_fcs").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("soilgasflux_fcs")
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
    return logger
