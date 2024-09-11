import threading
import os
import sys
from agent_management_system import AgentManagementSystem
import general_utils
import agent_creator
import functions
from voice_control import VoiceControl
from user_interface import UserInterface
from robotic_arm_assembly import RoboticArmAssembly
import json

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
    robot_init_list = general_utils.get_init_files(robot_file_path)
    # Spec File
    robot_spec_file = robot_file_path + 'specification/xArm_SOP.pdf'
    # Params File
    robot_params_file = robot_file_path + 'specification/params.json'
    robot_params_file2 = robot_file_path + 'specification/params2.json'
    # temp
    with open(robot_params_file2, 'r') as file:
        params_information = json.load(file)
    assembly = RoboticArmAssembly(params_information)

    # User Creation
    user = agent_creator.create_user()

    # Robot Agent Creation
    roboticarm_functions = functions.RoboticArmFunctions(robot_spec_file, robot_params_file)
    roboticarm_agents = agent_creator.create_robot_agents(robot_init_list, roboticarm_functions)
    agents_list = [user] + roboticarm_agents

    # Initialize the Agent Management System
    ams = AgentManagementSystem(agents=agents_list, mas_dir=".")

    # Start the Agent Management System in a separate thread
    ams_thread = threading.Thread(target=ams.thread_start, daemon=True)
    ams_thread.start()

    # Start the voice control in a separate thread
    voice_control_thread = threading.Thread(target=run_voice_control, args=(voice_control, user, roboticarm_agents), daemon=True)
    voice_control_thread.start()

    # Text-to-speech initialization message
    voice_control.text_to_speech("The xArm has been initiated.")

    # Start the UI in the main thread
    UserInterface.start_ui_loop()
