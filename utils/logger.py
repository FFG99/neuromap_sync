"""
Модуль для настройки логгера проекта
"""
import logging
import sys


def setup_logger(name: str = "neuromap_sync", level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера для проекта
    
    Args:
        name: Имя логгера
        level: Уровень логирования (по умолчанию INFO)
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    logger.setLevel(level)
    
    root_logger = logging.getLogger()
    if root_logger.level > level:
        root_logger.setLevel(level)
    
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Получить логгер для модуля
    
    Args:
        name: Имя модуля (если None, используется имя вызывающего модуля)
    
    Returns:
        Логгер
    """
    root_logger = logging.getLogger('neuromap_sync')
    if not root_logger.handlers:
        current_level = root_logger.level if root_logger.level != logging.NOTSET else logging.INFO
        setup_logger(level=current_level)
    
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'neuromap_sync')
    
    child_logger = logging.getLogger('neuromap_sync').getChild(name.split('.')[-1])
    child_logger.propagate = True
    child_logger.setLevel(logging.NOTSET)
    
    return child_logger

