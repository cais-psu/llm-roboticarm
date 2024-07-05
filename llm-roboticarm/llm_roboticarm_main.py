import tkinter as tk
from tkinter import scrolledtext
import threading
import time
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
        time.sleep(2)
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

# Function to handle detected hotword
def hotword_detected():
    log_message("Hotword 'grapefruit' detected!")
    start_recording()
    # You can add a sleep period to avoid immediate re-triggering if necessary
    time.sleep(5)  # Adjust the sleep duration as needed
    stop_recording()

# Function to continuously listen for hotword
def listen_for_hotword(porcupine, audio_stream):
    log_message("Listening for 'grapefruit' hotword...")
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            hotword_detected()

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

    # Initialize Porcupine with access key
    access_key = "FrMaUJNG+1dzKVWOW4J06mE81bkd6ao6vseEBG5iJ2AeaLqp/gFqIQ=="  # Replace with your Picovoice access key

    porcupine = pvporcupine.create(access_key=access_key, keywords=["grapefruit"])

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    ################################################# UI Configuration ###########################################################
    # Create the UI
    root = tk.Tk()
    root.title("Robotic Arm Voice Control")
    root.geometry("500x300")  # Adjusted the window size to accommodate the message log

    # Message log
    message_log = scrolledtext.ScrolledText(root, height=10, width=50, state='disabled', bg='white')
    message_log.pack(padx=10, pady=10)

    # Start listening in a separate thread
    threading.Thread(target=listen_for_hotword, args=(porcupine, audio_stream), daemon=True).start()

    root.mainloop()
    ##############################################################################################################################

    # Ensure the main thread doesn't exit
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
        log_message("Terminated by user.")
