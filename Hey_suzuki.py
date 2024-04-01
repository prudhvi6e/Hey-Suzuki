import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import speech_recognition as sr
import base64
import wave
from tools.hey_suzuki import suzuki_RAG
from dataclasses import dataclass
from st_audiorec import st_audiorec
# Set the title for the Streamlit app
# Create a file uploader in the sidebar
import os 

from elevenlabs import save
from elevenlabs.client import ElevenLabs

# Your existing text-to-speech function
def text_to_speech(text):
    # Enter your elevenlabs API key
    client = ElevenLabs(api_key="")
    audio = client.generate(
            text= text,
            voice="Rachel",
            model="eleven_multilingual_v2")
    save(audio, "audio/elevenlabs_output.mp3")
    return audio

def audio_to_text():
    r = sr.Recognizer()
    audio = sr.AudioFile("audio/output.wav")
    with audio as source:
        audio = r.record(source)                  
        result = r.recognize_google(audio)
    return result

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

@dataclass
class Message:
    actor: str
    payload: str

def main():
    with st.sidebar:
        st.image("images/MSIL_logo.png")
        wav_audio_data = st_audiorec()
        result = None 
        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')
            raw_audio_data  = wav_audio_data
            
            channels = 2  
            sample_width = 2  
            frame_rate = 44100  

            with wave.open('audio/output.wav', 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(frame_rate)
                wav_file.writeframes(raw_audio_data)

            result = audio_to_text()
            with st.expander("Speech to Text"):
                if result is not "None":
                    st.markdown("**Result:**")
                    st.write(result)
                else:
                    st.write("Please record your command")

    add_bg_from_local('images/Vitara_dashboard.png')    

    if result is not None:
        output = suzuki_RAG()
        answer = output.conversational_chat(result,top_k = 4,tempr = 0.1)
        audio = text_to_speech(answer)
        st.audio("audio/elevenlabs_output.mp3" , format = "audio/mp3")
    
    
if __name__=="__main__":
    main()