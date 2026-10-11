"""Sistema de logging mejorado."""

import logging
import sys


def setup_logger(name, level=logging.INFO):
    """Configura logger con formato consistente.
    
    Args:
        name: Nombre del logger (normalmente __name__)
        level: Nivel de logging (por defecto INFO)
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Solo agregar handler si no existe (evitar duplicados)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
