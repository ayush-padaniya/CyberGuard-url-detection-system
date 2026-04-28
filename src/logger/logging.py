import logging
import os
from datetime import datetime


# ==============================================
#           Creates the log folder
# ==============================================

os.makedirs('logs', exist_ok=True)


# ==============================================
#                Setting up logs format
# ==============================================

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join('logs', LOG_FILE)


# ==============================================
#                Configure Logging
# ==============================================

logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # save to file
        logging.FileHandler(LOG_FILE_PATH),
        # also print to terminal
        logging.StreamHandler()
    ]
)

# create logger
logger = logging.getLogger(__name__)