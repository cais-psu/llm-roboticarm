import tkinter as tk
from tkinter import scrolledtext
import threading
from voice_recorder import VoiceRecorder
import audio_utils
import time

import asyncio
from agent_management_system import AgentManagementSystem
import utils
import agent_creator
import functions

def log_message(message):
    message_log.configure(state='normal')  # Enable editing of the Text widget
    message_log.insert(tk.END, message + "\n")  # Append the message
    message_log.configure(state='disabled')  # Disable editing of the Text widget
    message_log.yview(tk.END)  # Auto-scroll to the bottom

def start_recording():
    if recorder.start_recording():
        threading.Thread(target=recorder.record_audio).start()
        time.sleep(2)
        log_message("Recording started!")
    else:
        log_message("Recording is already in progress.")

def stop_recording():
    if recorder.stop_recording():
        log_message("Recording stopped and saved successfully.")
        # Initialize the transcriber with your API key
        # Start transcription in a background thread to avoid blocking the UI
        threading.Thread(target=lambda: transcribe_and_append_command('voice_command.wav')).start()
    else:
        log_message("Recording is not active.")

def transcribe_and_append_command(audio_path):
    transcript = audio_utils.transcribe(audio_path)
    if transcript:
        log_message(f"Transcription: {transcript}")
        user.command.append(transcript)  # Appends the transcript to the command list
    else:
        log_message("Transcription failed or returned no result.")

recorder = VoiceRecorder()

# Define the file paths for the JSON files
robot_file_path = 'initialization/robots/'

# Init Files
robot_init_list = utils.get_init_files(robot_file_path)

# User Creation
user = agent_creator.create_user()

# Robot Agent Creation
roboticarm_functions = functions.RoboticArmFunctions(robot_init_list)
roboticarm_agents = agent_creator.create_robot_agents(robot_init_list, roboticarm_functions)

agents_list = [user] + roboticarm_agents

ams = AgentManagementSystem(agents=agents_list, mas_dir=".")

# Start the manufacturing system
ams.thread_start()

################################################# UI Configuration ###########################################################
# Create the UI
root = tk.Tk()
root.title("Robotic Arm Voice Control")
root.geometry("500x300")  # Adjusted the window size to accommodate the message log

button_font = ("Arial", 14)

start_button = tk.Button(root, text="Start Recording", command=start_recording, font=button_font, height=2, width=20)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, font=button_font, height=2, width=20)
stop_button.pack(pady=10)

# Message log
message_log = scrolledtext.ScrolledText(root, height=10, width=50, state='disabled', bg='white')
message_log.pack(padx=10, pady=10)

root.mainloop()
##############################################################################################################################