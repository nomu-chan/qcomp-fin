import logging, os, sys
from logging import Logger

def setup_logging(level: int = logging.INFO):
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
  logging.getLogger("urllib3").setLevel(logging.WARNING)
  logging.getLogger("requests").setLevel(logging.WARNING)
    # This specifically targets the font manager and keeps it quiet
  logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
  # 2. Silence the PNG Plugin (Pillow/PIL internal)
  logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

  # 3. Optional: Catch-all for PIL/Matplotlib noise
  logging.getLogger('PIL').setLevel(logging.WARNING)
  # If you want to silence all of Matplotlib's internal noise
  logging.getLogger('matplotlib').setLevel(logging.WARNING)
  logging.getLogger('')
  return logging.getLogger(name)