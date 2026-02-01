import logging
import sys
from typing import Optional
from src.config import LOGS_DIR

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger configurado con formato estándar.
    
    Args:
        name: Nombre del logger (usualmente __name__).
        level: Nivel de logging (default: logging.INFO).
        log_file: Ruta opcional para guardar logs en archivo.
    
    Returns:
        logging.Logger configurado.
    """
    logger = logging.getLogger(name)
    
    # Si ya tiene handlers, no agrega más para evitar duplicados
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler 
    # Si no se especifica archivo, usar el default del proyecto
    if log_file is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = str(LOGS_DIR / "app.log")

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
