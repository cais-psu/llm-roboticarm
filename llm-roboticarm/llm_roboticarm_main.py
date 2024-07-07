import tkinter as tk
from tkinter import scrolledtext
import threading
import audio_utils
import os
import sys
from agent_management_system import AgentManagementSystem
import robot_utils
import agent_creator
import functions
from voice_recorder import VoiceRecorder
import pvporcupine
import pyaudio
import struct

# Function to log messages in the UI
def log_message(message):
    message_log.configure(state='normal') 
    message_log.insert(tk.END, message + "\n")  
    message_log.configure(state='disabled')  
    message_log.yview(tk.END) 

# Function to start recording
def start_recording():
    if recorder.start_recording():
        threading.Thread(target=recorder.record_audio).start()
        log_message("Recording started!")
    else:
        log_message("Recording is already in progress.")

# Function to stop recording and transcribe
def stop_recording():
    if recorder.stop_recording():
        log_message("Recording stopped and saved successfully.")
        threading.Thread(target=lambda: transcribe_and_append_command('voice_command.wav')).start()
    else:
        log_message("Recording is not active.")

# Function to transcribe and append command
def transcribe_and_append_command(audio_path):
    transcript = audio_utils.transcribe(audio_path)
    if transcript:
        log_message(f"Transcription: {transcript}")
        user.command.append(transcript)  # Append the transcript to the command list
    else:
        log_message("Transcription failed or returned no result.")

# Function to handle detected start hotword
def start_hotword_detected():
    log_message("Hotword 'hello xarm' detected!")
    start_recording()

# Function to handle detected stop hotword
def stop_hotword_detected():
    log_message("Hotword 'end of command' detected!")
    stop_recording()

# Function to continuously listen for hotwords
def listen_for_hotwords(porcupine_start, porcupine_stop, audio_stream):
    log_message("Listening for hotwords 'hello xarm' and 'end of command'...")
    while True:
        pcm = audio_stream.read(porcupine_start.frame_length)
        pcm = struct.unpack_from("h" * porcupine_start.frame_length, pcm)
        
        start_keyword_index = porcupine_start.process(pcm)
        stop_keyword_index = porcupine_stop.process(pcm)
        
        if start_keyword_index >= 0:
            start_hotword_detected()
        elif stop_keyword_index >= 0:
            stop_hotword_detected()

if __name__ == "__main__":
    # Initialize the voice recorder
    recorder = VoiceRecorder()

    # Define the file paths for the JSON files
    robot_file_path = 'llm-roboticarm/initialization/robots/'

    # Init Files
    robot_init_list = robot_utils.get_init_files(robot_file_path)

    # User Creation
    user = agent_creator.create_user()

    # Robot Agent Creation
    roboticarm_functions = functions.RoboticArmFunctions(robot_init_list)
    roboticarm_agents = agent_creator.create_robot_agents(robot_init_list, roboticarm_functions)

    agents_list = [user] + roboticarm_agents

    # Initialize the Agent Management System
    ams = AgentManagementSystem(agents=agents_list, mas_dir=".")
    ams.thread_start()
    audio_utils.text_to_speech("The robot has been initiated.")

    # Initialize Porcupine with access key and custom keywords
    access_key = "FrMaUJNG+1dzKVWOW4J06mE81bkd6ao6vseEBG5iJ2AeaLqp/gFqIQ=="  # Replace with your Picovoice access key
    start_keyword_path = "llm-roboticarm/voice_keywords/hello_xarm_wakeword.ppn"  # Path to your custom start keyword model
    stop_keyword_path = "llm-roboticarm/voice_keywords/end_of_command_wakeword.ppn"  # Path to your custom stop keyword model

    porcupine_start = pvporcupine.create(access_key=access_key, keyword_paths=[start_keyword_path])
    porcupine_stop = pvporcupine.create(access_key=access_key, keyword_paths=[stop_keyword_path])

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine_start.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine_start.frame_length
    )

    ################################################# UI Configuration ###########################################################
    # Create the UI
    root = tk.Tk()
    root.title("Robotic Arm Voice Control")
    root.geometry("500x300")  # Adjusted the window size to accommodate the message log

    # Message log
    message_log = scrolledtext.ScrolledText(root, height=18, width=50, state='disabled', bg='white')
    message_log.pack(padx=10, pady=10)

    # Start listening in a separate thread
    threading.Thread(target=listen_for_hotwords, args=(porcupine_start, porcupine_stop, audio_stream), daemon=True).start()

    root.mainloop()
    ##############################################################################################################################

    # Clean up resources when the UI is closed
    audio_stream.close()
    pa.terminate()
    porcupine_start.delete()
    porcupine_stop.delete()
    log_message("Terminated by user.")


    access_key = "FrMaUJNG+1dzKVWOW4J06mE81bkd6ao6vseEBG5iJ2AeaLqp/gFqIQ=="  # Replace with your Picovoice access key
