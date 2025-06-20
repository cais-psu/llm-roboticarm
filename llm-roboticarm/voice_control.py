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
import numpy as np

class VoiceControl:
    """
    A class to handle voice control functionalities, including sound detection,
    recording audio, transcribing speech to text, and playing text-to-speech output.
    """    

    def __init__(self, target_device_name="CMTECK"):
        """
        Initializes the VoiceControl class with sound detection and audio recording setup.
        """
        self.target_device_name = target_device_name
        self.audio_interface = pyaudio.PyAudio()
        self.device_index = self.get_device_index()
        self.chunk_size = 1024  # Audio chunk size
        self.frame_length = 512  # Default frame length for PvRecorder
        self.recording = False
        self.audio = []
        self.lock = threading.Lock()
        self.recording_event = threading.Event()  # Event to signal recording completion

    def get_device_index(self):
        """
        Get the device index for the target microphone.

        Returns:
        int: The index of the microphone device.
        """
        device_count = self.audio_interface.get_device_count()
        for i in range(device_count):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if self.target_device_name.lower() in device_info["name"].lower():
                print(f"Found target device: {device_info['name']} (Index: {i})")
                return i
        raise ValueError(f"Target device '{self.target_device_name}' not found.")

    def monitor_microphone_activity(self, start_callback, stop_callback, threshold=500, sound_start_threshold=3, silence_start_threshold=5):
        """
        Monitors the microphone input for activity and triggers callbacks.

        Parameters:
        start_callback (function): Function to call when sound is detected.
        stop_callback (function): Function to call when no significant sound is detected.
        threshold (int): Minimum audio signal amplitude to consider as "sound detected".
        sound_start_threshold (int): Frames required to confirm sound start.
        silence_start_threshold (int): Frames required to confirm silence start.
        """
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        print("Monitoring microphone for activity...")

        sound_counter = 0
        silence_counter = 0

        try:
            while True:
                data = np.frombuffer(stream.read(self.chunk_size), dtype=np.int16)
                amplitude = np.abs(data).mean()  # Calculate average amplitude

                if amplitude > threshold:
                    sound_counter += 1
                    silence_counter = 0
                else:
                    silence_counter += 1
                    sound_counter = 0

                if sound_counter >= sound_start_threshold and not self.recording:
                    start_callback()
                    sound_counter = 0  # Reset counter after starting recording
                elif silence_counter >= silence_start_threshold and self.recording:
                    stop_callback()
                    silence_counter = 0  # Reset counter after stopping recording
        except KeyboardInterrupt:
            print("Stopping microphone monitoring.")
        finally:
            stream.stop_stream()
            stream.close()

    def transcribe(self, audio_file_path):
        """
        Transcribes audio to text using OpenAI's Whisper model.

        Parameters:
        audio_file_path (str): Path to the audio file to be transcribed.

        Returns:
        str or None: The transcription text if successful, None otherwise.
        """        
        try:
            # Check the audio file duration
            with wave.open(audio_file_path, 'rb') as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
                if duration < 0.1:
                    raise ValueError("Audio file is too short. Minimum audio length is 0.1 seconds.")

            # Proceed with transcription if duration is sufficient
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                return transcript
        except ValueError as ve:
            print(f"Failed to transcribe audio: {str(ve)}")
        except Exception as e:
            print(f"Failed to transcribe audio: {str(e)}")
        return None

    def text_to_speech(self, message, delay=0):
        """
        Converts text to speech and plays it through the system's audio output.
        """
        time.sleep(delay)  # Use time.sleep for blocking sleep
        file_path = "response.mp3"

        # Lock for thread safety
        with self.lock:
            # Check if recording is in progress
            if self.recording:
                print("Recording is in progress. Aborting speech playback.")
                return

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
                # Check again before playing
                if not self.recording:
                    self.play_mp3(file_path)
                else:
                    print("Recording started during speech synthesis. Aborting playback.")
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
        Waits for enough frames to be recorded and ensures all sounds are captured before stopping.
        """
        self.audio = []  # Make sure this is at the very beginning
        recorder = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
        try:
            recorder.start()
            print("Recording started!!")
            self.audio = []
            silence_buffer = 0  # Buffer to wait for remaining sounds
            silence_threshold = 60  # Frames of silence to allow before stopping recording

            while self.recording or silence_buffer < silence_threshold:
                frame = recorder.read()
                self.audio.extend(frame)

                # Calculate the average amplitude of the frame
                amplitude = np.abs(np.array(frame)).mean()

                if amplitude > 500:  # Adjust threshold for sound detection
                    silence_buffer = 0  # Reset silence buffer when sound is detected
                else:
                    silence_buffer += 1  # Increment silence buffer when no sound is detected

            # Check if enough frames are recorded before saving
            if len(self.audio) < 1600:  # Example: ~0.1 seconds at 16kHz sample rate
                print("Recording too short, skipping save.")
                return

            print(f"Total frames recorded: {len(self.audio)}")
        finally:
            recorder.stop()
            recorder.delete()
            print("Recording stopped.")
            self.save_recording()
            self.recording_event.set()  # Signal that recording is complete

    def start_recording(self):
        """
        Starts recording audio in a separate thread if not already recording.

        Returns:
        bool: True if recording started, False if already recording.
        """
        if not self.recording:
            self.recording = True
            self.recording_event.clear()  # Clear the event before starting a new recording
            threading.Thread(target=self.record_audio).start()
            return True
        else:
            print("Recording is already in progress.")
            return False

    def stop_recording(self):
        """
        Stops the audio recording.
        """
        if self.recording:
            self.recording = False
            self.recording_event.wait()  # Wait for the recording thread to signal completion
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

        try:
            # Check if the file already exists
            if os.path.exists(path):
                os.remove(path)

            # Now save the new recording
            print(f"Saving recording to {path}")
            with wave.open(path, 'w') as f:
                f.setparams((1, 2, 16000, 0, "NONE", "NONE"))
                f.writeframes(struct.pack('<' + ('h' * len(self.audio)), *self.audio))
            print("Recording saved successfully.")
        except Exception as e:
            print(f"Failed to save recording: {e}")

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
            user_command = transcript
            log_message("User", user_command)
            user.commands.append(user_command)  # Append the cleaned transcript to the command list
            # Send the command to the robot agents
            for agent in roboticarm_agents:
                agent.message("user", user_command)
        else:
            log_message("System", "Transcription failed or returned no result.")

    def start_sound_detected(self, log_message):
        """
        Callback triggeresd when sound is detected.
        """
        log_message("System", "Sound detected on CMTECK microphone!")
        if self.start_recording():
            # Stop any ongoing speech playback
            self.stop_and_unload_mixer()
            log_message("System", "Recording started!")
        else:
            log_message("System", "Recording is already in progress.")


    def stop_sound_detected(self, log_message, user, roboticarm_agents):
        """
        Callback triggered when no significant sound is detected.
        """
        if self.stop_recording():
            log_message("System", "Recording stopped and saved successfully.")
            self.recording_event.wait()  # Wait for the recording to finish
            print("Recording event signaled.")  # Add this line
            self.transcribe_and_append_command('voice_command.wav', user, log_message, roboticarm_agents)
        else:
            log_message("System", "Recording is not active.")
