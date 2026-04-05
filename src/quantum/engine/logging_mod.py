import logging, os, sys
from logging import Logger

def setup_logging(level: int = logging.DEBUG):
  log_format = "[%(asctime)s]|[%(filename)s:%(lineno)d]: %(levelname)s - %(message)s"
  
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
  
  logging.basicConfig(
    level=level,
    format=log_format,
    force=True,
    stream=sys.stderr
  )
    
def get_logging(name: str, level: int = logging.DEBUG):
  setup_logging(level)
  return logging.getLogger(name)