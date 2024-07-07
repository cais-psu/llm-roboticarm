import threading
import os
import sys
from agent_management_system import AgentManagementSystem
import robot_utils
import agent_creator
import functions
from voice_control import VoiceControl
from user_interface import UserInterface

def run_voice_control(voice_control, user, roboticarm_agents):
    # Start listening for hotwords
    voice_control.listen_for_hotwords(
        lambda: voice_control.start_hotword_detected(UserInterface.log_message_to_ui),
        lambda: voice_control.stop_hotword_detected(UserInterface.log_message_to_ui, user, roboticarm_agents)
    )

if __name__ == "__main__":
    # Initialize the VoiceControl class
    voice_control = VoiceControl()

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
    voice_control.text_to_speech("The robot has been initiated.")

    # Start the voice control in a separate thread
    voice_control_thread = threading.Thread(target=run_voice_control, args=(voice_control, user, roboticarm_agents), daemon=True)
    voice_control_thread.start()

    # Start the UI in the main thread
    UserInterface.start_ui_loop()