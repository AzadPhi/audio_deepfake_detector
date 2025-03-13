### params.py

import os
import numpy as np


#### VARIABLES
DATA_SIZE = os.environ.get("DATA_SIZE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME_RAW_DATA = os.environ.get("BUCKET_NAME_RAW_DATA")
BUCKET_PROCESSED_DATA = os.environ.get("BUCKET_PROCESSED_DATA")

INSTANCE = os.environ.get("INSTANCE")

PATH_TO_RAW_DATA=f"/home/{os.environ.get('USER_NAME')}/audio_deepfake_detector/raw_data"
PATH_PROCESSED_DATA=f"/home/{os.environ.get('USER_NAME')}/audio_deepfake_detector/processed_data/music_preprocessed_{os.environ.get('DATA_SIZE')}.csv"



#### CONSTANTS
SAMPLING_RATE = 16000  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
DURATION = 10  # Durée cible en secondes
N_MELS = 128  # Nombre de bandes de Mel

HOP_LENGTH = 347 * DURATION  # Détermine le nombre de frames temporelles
FMIN = 20  # Fréquence minimale
FMAX = SAMPLING_RATE // 2  # Fréquence maximale (Nyquist)
N_FFT = N_MELS * 20  # Taille de la FFT
SAMPLES = SAMPLING_RATE * DURATION
