
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
import pandas as pd

### ------------ Etape 1: Definition des paramètres ------------

class Conf:
    sampling_rate = 16000  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
    duration = 10  # Durée cible en secondes
    hop_length = 347 * duration  # Détermine le nombre de frames temporelles
    fmin = 20  # Fréquence minimale
    fmax = sampling_rate // 2  # Fréquence maximale (Nyquist)
    n_mels = 128  # Nombre de bandes de Mel
    n_fft = n_mels * 20  # Taille de la FFT
    samples = sampling_rate * duration  # Nombre total d'échantillons

conf = Conf()

### ------------ Etape 2: Première fonction pour Lecture et nettoyage de l'audio ------------

def read_audio(conf, pathname, trim_long_data=True):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate, mono=True)
    # Charge le fichier audio pathname avec une fréquence d'échantillonnage définie dans conf.sampling_rate (44100 Hz dans ce cas)
    # y est le signal audio sous forme d'un tableau NumPy
    # sr est la fréquence d'échantillonnage

    if 0 < len(y): # Évite une erreur en cas de fichier audio vide
        y, _ = librosa.effects.trim(y) # Supprime les silences au début et à la fin
    # La fonction librosa.effects.trim(y) supprime les parties silencieuses en début et fin d'audio

    # Si l'audio est plus long que la durée cible, on coupe l'excédent.
    # Si l'audio est trop court, on ajoute du padding (remplissage avec des zéros) des deux côtés pour uniformiser la taille.

    if len(y) > conf.samples: # Si l'audio est trop long
        if trim_long_data:
            y = y[0:0+conf.samples] # On garde seulement la partie nécessaire

    else: # Si l'audio est trop court, on ajoute du padding
        padding = conf.samples - len(y)    # Nombre d'échantillons à ajouter
        offset = padding // 2 # Ajout équilibré à gauche et à droite
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y

### ------------ Etape 3: Conversion en spectrogramme ------------

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

    # Cette fonction génère un mel-spectrogram (une version transformée du spectrogramme classique) dans un premier temps
    # Puis conversion en dB (plus représentatif de la perception humaine du son)
    # Enfin, conversion en float32 pour optimiser la mémoire et la compatibilité

### ------------ Etape 4: Réunion des deux fonctions ------------

def read_as_melspectrogram(conf, pathname, trim_long_data=False):
    x = read_audio(conf, pathname, trim_long_data)
    prep_results_arr = audio_to_melspectrogram(conf, x)
    return prep_results_arr

# -> la fonction permet d'obtenir des np array
# exemple array_test = read_as_melspectrogram(conf,music_1_path,trim_long_data=False)

### ------------ Etape 5 (OPTIONNEL): Fonction pour plotter notre résultat ressorti par l'étape 4 ------------

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

# -> la fonction permet d'obtenir des plots
# exemple array_test = plot_spectrogram(array_test,sr=conf.sampling_rate,hop_length=conf.hop_length,y_axis="log")

### ------------ Etape 6 : Convertit le np array et l'ajoute à un Dataframe ------------

def create_spectrogram_dataframe(conf, pathnames : list, trim_long_data=False):

    data = []

    for pathname in pathnames:
        music_id = pathname.split('/')[-1] # Extrait le nom de la musique
        folder_name = "/".join(pathname.split("/")[:-1]) # Extrait le lien / path
        prep_results_arr = read_as_melspectrogram(conf, pathname)  # Intègre l'array

        data.append([music_id, folder_name, prep_results_arr])

    df = pd.DataFrame(data, columns=["music_id", "folder_name", "music_array"])
    return df

def create_csv(df):
    df.to_csv('music_preprocessed.csv', index=True)

if __name__ == '__main__':
    pass
