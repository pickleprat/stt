from transformers import pipeline 
from tempfile import NamedTemporaryFile 
from typing import Dict 

import streamlit as st 
import librosa 

# you can make the user elect these models 
class Models: 
    latest: str = "openai/whisper-large-v3" 
    large: str = "openai/whisper-large" 
    medium: str = "openai/whisper-medium" 
    small: str = "openai/whisper-small" 
    base: str = "openai/whisper-base" 
    # write the instruction that smaller models mean compromise in quality to gain the output in lesser time 
st.header("Your audio application for my beautiful wife!") 

# you can alter the name of the model 
model_name: str = Models.base #TODO: potenial idea, mention name of the model before you deploy it 
output: Dict[str, str] 

audio = st.file_uploader("Upload file here my beautiful wife (make sure its an mp3 darling)", type=["mp3"])
button = st.button("Upload")    

if audio and button: 
    with NamedTemporaryFile(suffix=".mp3") as tmp: 
        tmp.write(audio.getvalue())
        tmp.seek(0) 
        whisper = pipeline("automatic-speech-recognition", model=model_name)
        audio, sr = librosa.load(tmp.name, sr=16_000) 

        output = whisper(
          audio,
          return_timestamps=True,
          generate_kwargs = {"task": "transcribe"},
          chunk_length_s=30,
        )

        st.text_area(output["text"]) 

