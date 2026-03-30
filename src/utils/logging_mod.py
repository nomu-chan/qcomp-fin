import logging, os, sys
from logging import Logger

def setup_logging(level: int = logging.INFO):
  log_format = "[%(asctime)s]|[%(name)s]: %(levelname)s - %(message)s"
  
  if logging.getLogger().handlers: return
  
  logging.basicConfig(
    level=level,
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
  )
    
def get_logging(name: str, level: int = logging.INFO):
  setup_logging(level)
  return logging.getLogger(name)