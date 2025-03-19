import streamlit as st
import requests
import tempfile
import os


st.title("AI-Generated Music Detector")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)",
                                 type=["wav", "mp3"])

if uploaded_file:
    # save the uploaded file and get the path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=f"./temp/") as temp:
        temp.write(uploaded_file.read())
        temp_audio_path = temp.name # temporary path to the audio

    st.audio(temp_audio_path, format="audio/wav")

    # send the file_path to the api:
    response = requests.post("http://127.0.0.1:8000/predict",
                             json={"file_path": temp_audio_path})

    if response.status_code == 200:
        result = response.json()
        print("$$$$$")
        print(result)
        print("$$$$$")
        st.header(result)
        # st.write(f"**Confidence:** {result['confidence']:.2f}")
