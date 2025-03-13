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
SAMPLING_RATE = 16000  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
DURATION = 10.0  # Durée cible en secondes
N_MELTS = 128  # Nombre de bandes de Mel

HOP_LENGTH = 347 * DURATION  # Détermine le nombre de frames temporelles
FMIN = 20  # Fréquence minimale
FMAX = SAMPLING_RATE // 2  # Fréquence maximale (Nyquist)
N_FFT = N_MELTS * 20  # Taille de la FFT
SAMPLES = SAMPLING_RATE * DURATION
