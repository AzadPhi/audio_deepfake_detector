import streamlit as st
import requests
import tempfile
import os
from nos_paquets.sound_prep.params import *


st.title("AI-Generated Music Detector")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)",
                                 type=["wav", "mp3"])

if uploaded_file:
    # save the uploaded file and get the path:
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=f"./temp/") as temp:
    #     temp.write(uploaded_file.read())
    #     temp_audio_path = temp.name # temporary path to the audio

    st.audio(uploaded_file)

    # send the file_path to the api:


    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    print(SERVICE_URL)
    response = requests.post(f"{SERVICE_URL}/predict", files=files)

    # response = requests.post("http://127.0.0.1:8000/predict",
    #                          json={"file_path": temp_audio_path})

    if response.status_code == 200:
        result = response.json()
        print("$$$$$")
        print(result)
        print("$$$$$")
        st.header(result)

    else:
        print(response)
