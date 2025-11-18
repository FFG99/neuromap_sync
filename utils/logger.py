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
    
    # Если логгер уже настроен, возвращаем его
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Форматтер для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Получить логгер для модуля
    
    Args:
        name: Имя модуля (если None, используется имя вызывающего модуля)
    
    Returns:
        Логгер
    """
    # Убеждаемся, что корневой логгер настроен
    root_logger = logging.getLogger('neuromap_sync')
    if not root_logger.handlers:
        setup_logger()
    
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'neuromap_sync')
    
    return logging.getLogger('neuromap_sync').getChild(name.split('.')[-1])

