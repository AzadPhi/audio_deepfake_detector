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

PATH_TO_RAW_DATA=os.environ.get('PATH_TO_RAW_DATA', "audio_deepfake_detector/raw_data")
PATH_PROCESSED_DATA = os.path.join(
    os.environ.get("PATH_PROCESSED_DATA", "audio_deepfake_detector/processed_data"),
    f"music_preprocessed_{DURATION}sec.csv"
)

#### PATHS & DATA
TARGET = os.environ.get('TARGET_ENV', "local") # à modifier selon la data que l'on vient prendre

#--- LOCAL: target ='local'
LOCAL_PATH_TO_RAW_DATA= os.environ.get('LOCAL_PATH_TO_RAW_DATA', 'code/NicoTerli/99-Perso/data_processed_1000.csv') #propore à chacun
LOCAL_PATH_TO_RAW_DATA_PAU= os.environ.get('LOCAL_PATH_TO_RAW_DATA', 'code/AzadPhi/audio_deepfake_detector/data/1000_final.csv')
#--- CheckPoint_Result
LOCAL_PATH_SAVE_WEIGHT = os.environ.get('LOCAL_PATH_SAVE_WEIGHT', 'code/AzadPhi/audio_deepfake_detector/ModelCheckpoint/checkpoint.model.keras')
CLOUD_PATH_SAVE_WEIGHT = os.environ.get('CLOUD_PATH_SAVE_WEIGHT', 'gs://checkpoint_result/checkpoint.model.keras' )

#---Model
MODEL = os.environ.get("MODEL")
