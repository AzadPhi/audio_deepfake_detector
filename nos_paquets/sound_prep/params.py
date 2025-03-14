### params.py

import os
import numpy as np

#### CONSTANTS
SAMPLING_RATE = 16000  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
DURATION = 10  # Durée cible en secondes
N_MELS = 128 # Nombre de bandes de Mel

HOP_LENGTH = 347 * DURATION  # Détermine le nombre de frames temporelles
FMIN = 20  # Fréquence minimale
FMAX = SAMPLING_RATE // 2  # Fréquence maximale (Nyquist)
N_FFT = N_MELS * 20  # Taille de la FFT
SAMPLES = SAMPLING_RATE * DURATION


#### VARIABLES
DATA_SIZE = os.environ.get("DATA_SIZE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME_RAW_DATA = os.environ.get("BUCKET_NAME_RAW_DATA")
BUCKET_PROCESSED_DATA = os.environ.get("BUCKET_PROCESSED_DATA")

INSTANCE = os.environ.get("INSTANCE")

PATH_TO_RAW_DATA=os.environ.get('PATH_TO_RAW_DATA')
PATH_PROCESSED_DATA=f"{os.environ.get('PATH_PROCESSED_DATA')}music_processed_{DURATION}sec.csv"
