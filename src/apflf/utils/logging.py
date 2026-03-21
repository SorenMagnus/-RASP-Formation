"""日志工具。"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """获取统一格式的日志对象。"""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
