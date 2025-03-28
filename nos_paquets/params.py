
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
BUCKET_CHECKPOINT = os.environ.get("BUCKET_CHECKPOINT")

PATH_TO_RAW_DATA=os.environ.get('PATH_TO_RAW_DATA', "audio_deepfake_detector/raw_data")
PATH_PROCESSED_DATA =  os.environ.get("PATH_PROCESSED_DATA")



#### PATHS & DATA
TARGET = os.environ.get('TARGET_ENV', "local") # à modifier selon la data que l'on vient prendre

#--- LOCAL: target ='local'
LOCAL_PATH_TO_RAW_DATA= os.environ.get('LOCAL_PATH_TO_RAW_DATA', 'code/NicoTerli/99-Perso/data_processed_1000.csv') #propore à chacun
LOCAL_PATH_TO_RAW_DATA_PAU= os.environ.get('LOCAL_PATH_TO_RAW_DATA', 'code/AzadPhi/audio_deepfake_detector/data/1000_final.csv')

#--- CheckPoint_Result
LOCAL_PATH_SAVE_WEIGHT_HEAVY = os.environ.get('LOCAL_PATH_SAVE_WEIGHT_HEAVY', 'code/AzadPhi/audio_deepfake_detector/ModelCheckpoint/Heavy/checkpoint.model.keras')
LOCAL_PATH_SAVE_WEIGHT_LIGHT = os.environ.get('LOCAL_PATH_SAVE_WEIGHT_LIGHT', 'code/AzadPhi/audio_deepfake_detector/ModelCheckpoint/Light/checkpoint.model.keras')
CLOUD_PATH_SAVE_WEIGHT_HEAVY = os.environ.get('CLOUD_PATH_SAVE_WEIGHT_HEAVY', 'gs://checkpoint_result/Model_heavy/checkpoint.model.keras' )
CLOUD_PATH_SAVE_WEIGHT_LIGHT = os.environ.get('CLOUD_PATH_SAVE_WEIGHT_LIGHT', 'gs://checkpoint_result/Model_light/checkpoint.model.keras' )

LOCAL_REGISTRY_MODEL_PATH = os.environ.get('LOCAL_REGISTRY_MODEL_PATH')
LOCAL_PATH_TO_MODEL = os.environ.get('LOCAL_PATH_TO_MODEL')

#---Model
MODEL = os.environ.get("MODEL")

SERVICE_URL = os.environ.get("SERVICE_URL")
