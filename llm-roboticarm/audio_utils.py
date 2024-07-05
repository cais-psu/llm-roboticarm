import os
import openai
import time
import pygame

# load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

def transcribe(audio_file_path):
    try:
        time.sleep(1)
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text"
            )
            #print(transcript)
            return transcript
    except Exception as e:
        print(f"Failed to transcribe audio: {str(e)}")
        return None

def text_to_speech(message, delay=0):
    time.sleep(delay)
    file_path = "response.mp3"

    # Initialize Pygame mixer
    pygame.mixer.init()
    stop_and_unload_mixer()

    # Check if the file already exists and delete it
    if os.path.exists(file_path):
        time.sleep(1)
        os.remove(file_path)    

    try:
        soundoutput = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=message
        )
        soundoutput.stream_to_file("response.mp3")
        play_mp3("response.mp3")
    except Exception as e:
        print(f"Failed to convert from text to speech: {str(e)}")
        return None

def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def stop_and_unload_mixer():
    # Stop any currently playing music and unload it to free the file.
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
