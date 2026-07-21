"""日志配置，与 mildoc_index 保持一致风格"""
import logging


def setup_logging(name: str = 'mildoc_wxkf', level: int = logging.INFO) -> logging.Logger:
    """配置并返回应用日志器"""
    log_format = '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format, handlers=[logging.StreamHandler()])
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = setup_logging()
