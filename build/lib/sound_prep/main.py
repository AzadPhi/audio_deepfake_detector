from sound_prep.params import *
from sound_prep.preprocess import *
from pathlib import Path

### à rajouter dans le fichier main.py

from google.cloud.storage import Client, transfer_manager

# PATH_RAW_DATA = "../raw_data" # à modifier en fonction

def download_data_from_cloud(BUCKET_NAME, PATH_RAW_DATA, DATA_SIZE):
    """this function download the data stored on a bucket on gcloud, it stores is in the chosen directory
    if max_results==None, it downloads all the data"""

    client = Client()
    workers=2

    bucket = client.bucket(BUCKET_NAME)

    blob_names = [blob.name for blob in bucket.list_blobs(max_results=DATA_SIZE)]

    results = transfer_manager.download_many_to_path(bucket,
                                                     blob_names,
                                                     PATH_RAW_DATA=PATH_RAW_DATA,
                                                     max_workers=workers)
    print("The data has been downloaded!")

    directory = Path(PATH_RAW_DATA)
    files_path = [p for p in directory.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    string_paths = [str(path) for path in files_path]

    return string_paths

if __name__ == '__main__':
    string_paths = download_data_from_cloud(BUCKET_NAME=BUCKET_NAME,
                             PATH_RAW_DATA=PATH_RAW_DATA,
                             DATA_SIZE=DATA_SIZE)

    df = create_spectrogram_dataframe(conf, string_paths, trim_long_data=False)
    create_csv(df)
