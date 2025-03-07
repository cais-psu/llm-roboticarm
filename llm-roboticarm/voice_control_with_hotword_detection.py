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
    """
    A class to handle voice control functionalities, including hotword detection,
    recording audio, transcribing speech to text, and playing text-to-speech output.
    """    

    def __init__(self):
        """
        Initializes the VoiceControl class with hotword detection, OpenAI API key,
        and audio recording setup.
        """
        # Picovoice access key and paths to custom wakeword models

        self.access_key = "FrMaUJNG+1dzKVWOW4J06mE81bkd6ao6vseEBG5iJ2AeaLqp/gFqIQ=="  # Picovoice access key
        self.start_keyword_path = "llm-roboticarm/voice_keywords/hello_xarm_wakeword.ppn"  # Path to your custom start keyword model
        self.stop_keyword_path = "llm-roboticarm/voice_keywords/end_of_command_wakeword.ppn"  # Path to your custom stop keyword model
        self.lock = threading.Lock()

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
        """
        Transcribes audio to text using OpenAI's Whisper model.

        Parameters:
        audio_file_path (str): Path to the audio file to be transcribed.

        Returns:
        str or None: The transcription text if successful, None otherwise.
        """        
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
        """
        Converts text to speech and plays it through the system's audio output.

        Parameters:
        message (str): The text to convert to speech.
        delay (int, optional): Time delay in seconds before starting playback.
        """        
        time.sleep(delay)  # Use time.sleep for blocking sleep
        file_path = "response.mp3"

        # Lock for thread safety
        with self.lock:
            # Initialize Pygame mixer
            pygame.mixer.init()
            self.stop_and_unload_mixer()

            # Check if the file already exists and delete it
            if os.path.exists(file_path):
                os.remove(file_path)

            try:
                soundoutput = openai.audio.speech.create(
                    model="tts-1",
                    voice="onyx",
                    input=message
                )
                soundoutput.stream_to_file(file_path)
                self.play_mp3(file_path)
                        
            except Exception as e:
                print(f"Failed to convert from text to speech: {str(e)}")
                return None
        
    def play_mp3(self, file_path):
        """
        Plays an MP3 file using Pygame.

        Parameters:
        file_path (str): Path to the MP3 file to play.
        """        
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    def stop_and_unload_mixer(self):
        """
        Stops any audio currently playing and unloads the Pygame mixer.
        """        
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()

    def record_audio(self):
        """
        Records audio using PvRecorder and stores it in the audio list.
        """        
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
        """
        Starts recording audio in a separate thread if not already recording.

        Returns:
        bool: True if recording started, False if already recording.
        """        
        if not self.recording:
            self.recording = True
            threading.Thread(target=self.record_audio).start()
            return True
        else:
            print("Recording is already in progress.")
            return False

    def stop_recording(self):
        """
        Stops the audio recording.

        Returns:
        bool: True if recording was active and stopped, False otherwise.
        """        
        if self.recording:
            self.recording = False
            return True
        else:
            print("Recording is not active.")
            return False

    def save_recording(self):
        """
        Saves the recorded audio to a WAV file.
        """        
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
        """
        Listens for start and stop hotwords using Porcupine and triggers respective callbacks.

        Parameters:
        start_hotword_callback (function): Callback for when the start hotword is detected.
        stop_hotword_callback (function): Callback for when the stop hotword is detected.
        """        
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
        """
        Transcribes audio, logs the command, and sends it to robotic arm agents.

        Parameters:
        audio_path (str): Path to the audio file.
        user: The user object for appending the command.
        log_message (function): Function to log messages to the UI.
        roboticarm_agents (list): List of robotic arm agent instances to receive the command.
        """        
        transcript = self.transcribe(audio_path)
        if transcript:
            # Remove the last three words from the transcription
            #user_command = ' '.join(transcript.split()[:-3])
            user_command = transcript
            log_message("User", user_command)
            user.commands.append(user_command)  # Append the cleaned transcript to the command list
            # Send the command to the robot agents
            for agent in roboticarm_agents:
                agent.message("user", user_command)
        else:
            log_message("System", "Transcription failed or returned no result.")

    def start_hotword_detected(self, log_message):
        """
        Callback triggered when the start hotword is detected.

        Parameters:
        log_message (function): Function to log messages to the UI.
        """        
        log_message("System", "Hotword 'hello xarm' detected!")
        if self.start_recording():
            log_message("System", "Recording started!")
        else:
            log_message("System", "Recording is already in progress.")

    def stop_hotword_detected(self, log_message, user, roboticarm_agents):
        """
        Callback triggered when the stop hotword is detected.

        Parameters:
        log_message (function): Function to log messages to the UI.
        user: The user object for appending the command.
        roboticarm_agents (list): List of robotic arm agent instances to receive the command.
        """        
        log_message("System", "Hotword 'end of command' detected!")
        if self.stop_recording():
            log_message("System", "Recording stopped and saved successfully.")
            threading.Thread(target=lambda: self.transcribe_and_append_command('voice_command.wav', user, log_message, roboticarm_agents)).start()
        else:
            log_message("System", "Recording is not active.")
