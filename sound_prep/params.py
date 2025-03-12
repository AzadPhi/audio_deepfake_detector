### params.py

import os
import numpy as np


#### VARIABLES
DATA_SIZE = os.environ.get("DATA_SIZE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

PATH_TO_RAW_DATA=f'/home/{os.environ.get("USER_NAME")}/audio_deepfake_detector/raw_data'
PATH_PROCESSED_DATA=f'/home/{os.environ.get("USER_NAME")}/audio_deepfake_detector/processed_data'

#### CONSTANTS
