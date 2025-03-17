### params.py

import os
import numpy as np


#### CONSTANTS
SAMPLING_RATE = 16000  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
DURATION = 2  # Durée cible en secondes
N_MELS = 128 # Nombre de bandes de Mel

N_FFT = N_MELS * 4  # Taille de la FFT (ou N_MELS * 8, mais plus lourd)
HOP_LENGTH = N_FFT // 2 # Détermine le nombre de frames temporelles : plus c'est petit, plus c'est détaillé (j'ai hésité avec N_FFT // 4, plus lourd)
FMIN = 20  # Fréquence minimale
FMAX = SAMPLING_RATE // 2  # Fréquence maximale
SAMPLES = SAMPLING_RATE * DURATION

#### VARIABLES
DATA_SIZE = os.environ.get("DATA_SIZE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME_RAW_DATA = os.environ.get("BUCKET_NAME_RAW_DATA")
BUCKET_PROCESSED_DATA = os.environ.get("BUCKET_PROCESSED_DATA")

INSTANCE = os.environ.get("INSTANCE")

PATH_TO_RAW_DATA=os.environ.get('PATH_TO_RAW_DATA')
PATH_PROCESSED_DATA=os.environ.get('PATH_PROCESSED_DATA')
