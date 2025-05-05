import threading
import os
import json

from agent_management_system import AgentManagementSystem
import general_utils
import agent_creator
import functions

from voice_control import VoiceControl
from user_interface import UserInterface

from robot_controller import RobotController
from camera_manager import CameraManager
from robot_tasks import RobotTask

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

    # Define the file paths
    initialization_file_path = 'llm-roboticarm/initialization/'
    specification_file_path = 'llm-roboticarm/specification/'

    # Init Files
    robot_file_path = initialization_file_path + 'robots'
    robot_init_list = general_utils.get_init_files(robot_file_path)
    with open(initialization_file_path + "resources/robots/robots.json") as f:
        robot_config = json.load(f)["ur5e"]
    with open(initialization_file_path + "resources/sensors/camera.json") as f:
        camera_config = json.load(f)
    with open(initialization_file_path + "products/products.json") as f:
        product_config = json.load(f)

    # Spec File
    robot_spec_file = os.path.join(initialization_file_path, 'robots/specification/SOP.pdf')

    # Hardware Initiation
    robot_controller = RobotController(robot_config)
    camera_manager = CameraManager(camera_config)
    robot_task = RobotTask(robot_controller, camera_manager, robot_config, product_config)

    # User Creation
    user = agent_creator.create_user()

    # Initialize robotic arm functions using specification and general parameters
    roboticarm_functions = functions.RoboticArmFunctions(robot_spec_file, robot_config, robot_task)
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