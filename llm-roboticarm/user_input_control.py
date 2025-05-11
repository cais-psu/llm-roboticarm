
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
from elevenlabs import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

class UserInputControl:
    """
    A class to handle voice control functionalities, including sound detection,
    recording audio, transcribing speech to text, and playing text-to-speech output.
    """    

    def __init__(self, target_device_name="CMTECK"):
        """
        Initializes the User Input Control class with sound detection and audio recording setup.
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
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    '''
    def get_device_index2(self):
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
    '''

    def get_device_index(self):
        """
        Get the device index for the target microphone, or fall back to the system default.

        Returns:
        int: The index of the microphone device.
        """
        device_count = self.audio_interface.get_device_count()
        for i in range(device_count):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if self.target_device_name.lower() in device_info["name"].lower():
                print(f"Found target device: {device_info['name']} (Index: {i})")
                return i
        # Fallback to default input device
        try:
            default_index = self.audio_interface.get_default_input_device_info()["index"]
            print(f"Target mic not found. Using default input device (Index: {default_index})")
            return default_index
        except Exception as e:
            raise RuntimeError(f"Could not find a valid input device: {e}")

    def monitor_microphone_activity(self, start_callback, stop_callback, threshold=1000, sound_start_threshold=3, silence_start_threshold=5, stop_delay=3.0):
        """
        Monitors the microphone input for activity and triggers callbacks.

        Parameters:
        start_callback (function): Function to call when sound is detected.
        stop_callback (function): Function to call when no significant sound is detected.
        threshold (int): Minimum audio signal amplitude to consider as "sound detected".
        sound_start_threshold (int): Frames required to confirm sound start.
        silence_start_threshold (int): Frames required to confirm silence start.
        stop_delay (float): Additional delay (in seconds) to confirm silence before stopping recording.
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
        silence_detected_time = None  # Time when silence was first detected

        try:
            while True:
                data = np.frombuffer(stream.read(self.chunk_size, exception_on_overflow=False), dtype=np.int16)

                # Compute RMS for better accuracy
                # amplitude = np.sqrt(np.mean(np.square(data)))
                amplitude = np.abs(data).mean()  # Calculate average amplitude

                #print(f"Amplitude: {amplitude}")  # Debugging - Show real-time amplitude

                if amplitude > threshold:
                    sound_counter += 1
                    silence_counter = 0
                    silence_detected_time = None  # Reset silence timer
                else:
                    if silence_counter == 0:
                        silence_detected_time = time.time()  # Mark when silence starts
                    silence_counter += 1
                    sound_counter = 0

                # Start recording if enough sound frames are detected
                if sound_counter >= sound_start_threshold and not self.recording:
                    print("Voice detected - Starting Recording")
                    start_callback()
                    sound_counter = 0  # Reset counter after starting recording

                # Delay stopping to confirm silence
                elif silence_counter >= silence_start_threshold and self.recording:
                    if silence_detected_time and (time.time() - silence_detected_time) >= stop_delay:
                        print("Silence detected for long enough - Confirming stop...")
                        
                        # Re-check audio to confirm silence before stopping
                        confirm_data = np.frombuffer(stream.read(self.chunk_size, exception_on_overflow=False), dtype=np.int16)
                        confirm_amplitude = np.sqrt(np.mean(np.square(confirm_data)))

                        if confirm_amplitude < (threshold * 0.5):  # Extra confirmation of silence
                            print("Confirmed silence - Stopping recording")
                            stop_callback()
                            silence_counter = 0  # Reset counter after stopping recording
                            silence_detected_time = None  # Reset timer
                        else:
                            print("False alarm - Resuming monitoring")

        except KeyboardInterrupt:
            print("Stopping microphone monitoring.")
        finally:
            stream.stop_stream()
            stream.close()

    def transcribe(self, audio_file_path):
        """
        Transcribes audio to text using ElevenLabs' Scribe model.

        Parameters:
        audio_file_path (str): Path to the audio file to be transcribed.

        Returns:
        str or None: The transcription text if successful, None otherwise.
        """
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
                if duration < 0.1:
                    raise ValueError("Audio file is too short. Minimum audio length is 0.1 seconds.")

            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v1",
                    diarize=True,
                    timestamps_granularity="word",
                    tag_audio_events=True
                )
                return transcript.text  # ✅ Use attribute instead of .get()
        except ValueError as ve:
            print(f"Failed to transcribe audio: {str(ve)}")
        except Exception as e:
            print(f"Failed to transcribe audio: {str(e)}")
        return None


    def transcribe_openai(self, audio_file_path):
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
                    response_format="text",
                    temperature=0
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
                    voice="echo",
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

    def record_audio(self, rate=16000):
        """
        Records audio using PyAudio. Automatically stops after silence.
        No fixed length — ends based on silence detection.
        """
        self.audio = []
        try:
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index
            )

            print("Recording started...")
            silence_buffer = 0
            silence_threshold = 60  # number of silent chunks before stopping

            while self.recording or silence_buffer < silence_threshold:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = struct.unpack('<' + ('h' * self.chunk_size), data)
                self.audio.extend(audio_data)

                amplitude = np.abs(np.array(audio_data)).mean()

                if amplitude > 500:
                    silence_buffer = 0
                else:
                    silence_buffer += 1

            stream.stop_stream()
            stream.close()
            print("Recording stopped.")

            if len(self.audio) < 1600:
                print("Recording too short, skipping save.")
                return

            self.save_recording()

        except Exception as e:
            print(f"[UserInputControl] PyAudio recording error: {e}")
            self.audio = []

        finally:
            self.recording_event.set()

    '''
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
    '''

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

    def process_text_command(self, text_command, user, log_message, roboticarm_agents):
        """
        Handles user text input, logs it, and sends it to the robotic agents.

        Parameters:
        text_command (str): The command entered by the user.
        user: The user object to track commands.
        log_message (function): A function to update the GUI.
        roboticarm_agents (list): List of robotic arms to send the command to.
        """
        log_message("User", text_command)
        user.commands.append(text_command)
        for agent in roboticarm_agents:
            agent.message("user", text_command)