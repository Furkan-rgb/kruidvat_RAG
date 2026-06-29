import logging
import os
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

LOG_FILE = os.environ.get("KRUIDVAT_LOG", "kruidvat_extractor.log")


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


def get_logger(name="kruidvat"):
    """Return a configured logger that writes JSON lines to a rotating file.

    The logger is idempotent (calling multiple times won't add duplicate handlers).
    Callers should log JSON by passing json.dumps(dict) as the message.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    logger._configured = True
    return logger
