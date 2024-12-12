import threading
import os
from agent_management_system import AgentManagementSystem
import general_utils
import agent_creator
import functions
from voice_control import VoiceControl
from user_interface import UserInterface
from robotic_arm_assembly import RoboticArmAssembly
import json

def run_voice_control(voice_control, user, roboticarm_agents):
    """
    Starts the voice control system to monitor microphone activity and execute 
    appropriate actions upon sound detection.

    Args:
        voice_control (VoiceControl): The voice control instance for sound detection.
        user (User): The user agent interacting with the voice control.
        roboticarm_agents (list): List of robotic arm agent instances.
    """
    # Monitor microphone activity for sound detection
    voice_control.monitor_microphone_activity(
        lambda: voice_control.start_sound_detected(UserInterface.log_message_to_ui),
        lambda: voice_control.stop_sound_detected(UserInterface.log_message_to_ui, user, roboticarm_agents)
    )

if __name__ == "__main__":
    # Initialize the VoiceControl class for CMTECK microphone
    voice_control = VoiceControl(target_device_name="CMTECK")

    # Define the file paths for the JSON files
    robot_file_path = 'llm-roboticarm/initialization/robots/'
    # Init Files
    robot_init_list = general_utils.get_init_files(robot_file_path)

    # Specify paths for robot specification and parameter files
    robot_spec_file = os.path.join(robot_file_path, 'specification/xArm_SOP.pdf')
    params_general_path = os.path.join(robot_file_path, 'specification/params_general.json')
    params_movement_path = os.path.join(robot_file_path, 'specification/params_movement.json')
    
    with open(params_general_path, 'r') as file:
        params_general = json.load(file)

    with open(params_movement_path, 'r') as file:
        params_movement = json.load(file)

    # Initialize the RoboticArmAssembly with both parameters
    assembly = RoboticArmAssembly(params_general, params_movement)

    # User Creation
    user = agent_creator.create_user()

    # Initialize robotic arm functions using specification and general parameters
    roboticarm_functions = functions.RoboticArmFunctions(robot_spec_file, params_general, params_movement)
    roboticarm_agents = agent_creator.create_robot_agents(robot_init_list, roboticarm_functions)
    agents_list = [user] + roboticarm_agents

    # Start the AMS in a separate daemon thread to enable concurrent operation
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
