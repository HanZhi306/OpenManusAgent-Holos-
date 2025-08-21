import os
import logging
import logging.handlers
import threading
from datetime import datetime

inited = False

def thread_id_filter(record):
    """Inject thread_id to log records"""
    record.thread_id = threading.get_ident() % 100000
    return record

def init_log(log_dir="log", log_level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """Initialize logging configuration with automatic log rotation."""
    global inited

    if inited:
        return

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Setup handlers with rotation
    handlers = [
        # Rotating file handler - rotates when file reaches max_bytes
        logging.handlers.RotatingFileHandler(f"{log_dir}/main.log", maxBytes=max_bytes, backupCount=backup_count),
        logging.StreamHandler()
    ]

    # Add thread ID filter to all handlers
    for h in handlers:
        h.addFilter(thread_id_filter)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(thread_id)d %(name)s:%(lineno)d [%(levelname)s] %(message)s",
        handlers=handlers
    )

    # Set werkzeug logger to ERROR level to reduce noise
    werkzeug_log = logging.getLogger('werkzeug')
    werkzeug_log.setLevel(logging.ERROR)

    inited = True


def get_logger(name, log_dir="log", log_level=logging.INFO, rotation_type="size"):
    #log_level = logging.DEBUG
    global inited
    if not inited:
        init_log(log_dir, log_level)
    return logging.getLogger(name)
