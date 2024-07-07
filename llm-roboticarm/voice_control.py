import os
import openai
import time
import pygame
import threading
import pvporcupine
import pyaudio
import struct
import wave
from pvrecorder import PvRecorder

class VoiceControl:
    def __init__(self):
        # Initialize the VoiceControl class with access key and keyword paths
        self.access_key = "FrMaUJNG+1dzKVWOW4J06mE81bkd6ao6vseEBG5iJ2AeaLqp/gFqIQ=="  # Picovoice access key
        self.start_keyword_path = "llm-roboticarm/voice_keywords/hello_xarm_wakeword.ppn"  # Path to your custom start keyword model
        self.stop_keyword_path = "llm-roboticarm/voice_keywords/end_of_command_wakeword.ppn"  # Path to your custom stop keyword model

        # Load the OpenAI API key
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if self.OPENAI_API_KEY is None:
            raise ValueError("No API key for OpenAI found in the environment variables.")
        openai.api_key = self.OPENAI_API_KEY

        # Initialize the voice recorder
        self.device_index = 0
        self.frame_length = 512
        self.recording = False
        self.audio = []

        # Initialize Porcupine Platform
        self.porcupine_start = pvporcupine.create(access_key=self.access_key, keyword_paths=[self.start_keyword_path])
        self.porcupine_stop = pvporcupine.create(access_key=self.access_key, keyword_paths=[self.stop_keyword_path])
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.porcupine_start.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine_start.frame_length
        )

    def transcribe(self, audio_file_path):
        try:
            time.sleep(1)
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcript
        except Exception as e:
            print(f"Failed to transcribe audio: {str(e)}")
            return None

    def text_to_speech(self, message, delay=0):
        time.sleep(delay)
        file_path = "response.mp3"

        # Initialize Pygame mixer
        pygame.mixer.init()
        self.stop_and_unload_mixer()

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
            self.play_mp3("response.mp3")
        except Exception as e:
            print(f"Failed to convert from text to speech: {str(e)}")
            return None

    def play_mp3(self, file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def stop_and_unload_mixer(self):
        # Stop any currently playing music and unload it to free the file.
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

    def record_audio(self):
        recorder = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
        try:
            recorder.start()
            print("Recording started!!")
            self.audio = []
            while self.recording:
                frame = recorder.read()
                self.audio.extend(frame)
        finally:
            recorder.stop()
            recorder.delete()
            print("Recording stopped.")
            self.save_recording()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            threading.Thread(target=self.record_audio).start()
            return True
        else:
            print("Recording is already in progress.")
            return False

    def stop_recording(self):
        if self.recording:
            self.recording = False
            return True
        else:
            print("Recording is not active.")
            return False

    def save_recording(self):
        if not self.audio:
            print("No audio recorded.")
            return
        path = 'voice_command.wav'

        # Check if the file already exists
        if os.path.exists(path):
            os.remove(path)

        # Now save the new recording
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 0, "NONE", "NONE"))
            f.writeframes(struct.pack('<' + ('h' * len(self.audio)), *self.audio))
        print("Recording saved successfully.")

    def listen_for_hotwords(self, start_hotword_callback, stop_hotword_callback):
        while True:
            pcm = self.audio_stream.read(self.porcupine_start.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine_start.frame_length, pcm)

            start_keyword_index = self.porcupine_start.process(pcm)
            stop_keyword_index = self.porcupine_stop.process(pcm)

            if start_keyword_index >= 0:
                start_hotword_callback()
            elif stop_keyword_index >= 0:
                stop_hotword_callback()

    def transcribe_and_append_command(self, audio_path, user, log_message, roboticarm_agents):
        transcript = self.transcribe(audio_path)
        if transcript:
            # Remove the last three words from the transcription
            user_command = ' '.join(transcript.split()[:-3])
            log_message("User", user_command)
            user.command.append(user_command)  # Append the cleaned transcript to the command list
            # Send the command to the robot agents
            for agent in roboticarm_agents:
                agent.message("user", user_command)
        else:
            log_message("System", "Transcription failed or returned no result.")

    def start_hotword_detected(self, log_message):
        log_message("System", "Hotword 'hello xarm' detected!")
        if self.start_recording():
            log_message("System", "Recording started!")
        else:
            log_message("System", "Recording is already in progress.")

    def stop_hotword_detected(self, log_message, user, roboticarm_agents):
        log_message("System", "Hotword 'end of command' detected!")
        if self.stop_recording():
            log_message("System", "Recording stopped and saved successfully.")
            threading.Thread(target=lambda: self.transcribe_and_append_command('voice_command.wav', user, log_message, roboticarm_agents)).start()
        else:
            log_message("System", "Recording is not active.")
