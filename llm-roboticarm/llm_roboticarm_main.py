import threading
import os
import json

from agent_management_system import AgentManagementSystem
import general_utils
import agent_creator
import functions

from user_input_control import UserInputControl
from user_interface import UserInterface

from robot_controller import RobotController
from camera_manager import CameraManager
from robot_tasks import RobotTask

def run_voice_control(voice_control, user, roboticarm_agents):
    """
    Starts the voice control system to monitor microphone activity and execute 
    appropriate actions upon sound detection.
    """
    voice_control.monitor_microphone_activity(
        lambda: voice_control.start_sound_detected(UserInterface.log_message_to_ui),
        lambda: voice_control.stop_sound_detected(UserInterface.log_message_to_ui, user, roboticarm_agents)
    )

if __name__ == "__main__":
    # === Initialize Human Input ===
    user_input_control = UserInputControl(target_device_name="CMTECK")

    # === Load Configuration Files ===
    initialization_file_path = 'llm-roboticarm/initialization/'
    specification_file_path = 'llm-roboticarm/specification/'

    with open(initialization_file_path + "resources/robots/robots.json") as f:
        robot_config = json.load(f)["ur5e"]
    with open(initialization_file_path + "resources/sensors/camera.json") as f:
        camera_config = json.load(f)
    with open(initialization_file_path + "products/products.json") as f:
        product_config = json.load(f)

    robot_spec_file = os.path.join(specification_file_path, 'SOP.pdf')

    # === Initialize Hardware Control Classes ===
    robot_controller = RobotController(robot_config)
    camera_manager = CameraManager(camera_config)
    robot_task = RobotTask(robot_controller, camera_manager, robot_config, product_config)

    # === Create Agents ===
    user = agent_creator.create_user()
    roboticarm_functions = functions.RoboticArmFunctions(robot_spec_file, robot_config, product_config, robot_task)

    # create robot agents
    robot_file_path = initialization_file_path + 'resources/robots/'
    robot_init_list = general_utils.get_init_files(robot_file_path)
    roboticarm_agents = agent_creator.create_robot_agents(robot_init_list, roboticarm_functions)

    # all agents list in the system
    agents_list = [user] + roboticarm_agents

    # === Start Agent Management System (AMS) Thread ===
    ams = AgentManagementSystem(agents=agents_list, mas_dir=".")
    ams_thread = threading.Thread(target=ams.thread_start, daemon=True)
    ams_thread.start()

    # === Start Voice Control Thread ===
    user_input_control_thread = threading.Thread(
        target=run_voice_control, 
        args=(user_input_control, user, roboticarm_agents), 
        daemon=True
    )
    user_input_control_thread.start()

    # === Text-to-speech Startup Message ===
    user_input_control.text_to_speech("UR5e has been initiated.")

    # === Initialize and Start GUI (Main Thread) ===
    # After all agents and control classes are initialized
    ui = UserInterface.get_instance()
    ui.user_input_control = user_input_control
    ui.user = user
    ui.roboticarm_agents = roboticarm_agents
    ui.start_ui()