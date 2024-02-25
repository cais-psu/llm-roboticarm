import robot_utils
import sys, os
# Add the HVAC_Assembly.py directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))

import robotic_arm_assembly

class RoboticArmFunctions:
    def __init__(self, init_list):
        self.configs = self.load_all_resource_data(init_list)
        self.loggers = {}

    @staticmethod
    def load_all_resource_data(init_list):
        all_configs = {}
        for init in init_list:
            config_data = robot_utils.load_json_data(init)
            all_configs.update(config_data)
        return all_configs

    def sorting(self, robot_name: str) -> dict:
        """
        This is a sorting function. When executed the product is sorted.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that sorts the product
        """
        print(f"Sorting product {robot_name}")
        
        result = {
            "func_type": "roboticarm_process",
            "robot_name": robot_name,
            "status": "completed",
            "content": f"Sorting by {robot_name} is successfully completed." 
        }
        return result
    
    def assembly(self, robot_name: str) -> dict:
        """
        This is an assembly function. When executed, the product is assemblied.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that assembles the product
        """
        assembly = robotic_arm_assembly.RoboticArmAssembly()
        step_completed, message = assembly.start_robotic_assembly()

        result = {
            "func_type": "roboticarm_process",
            "robot_name": robot_name,
            "step_completed": step_completed,
            "content": message 
        }
            
        return result   
    
    def resume_assembly(self, robot_name: str, step_completed: str) -> dict:
        """
        This is a resuming assembly function. When executed, the product will do assembly from the point where it left off.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that resumes the assembly of the product
        :param step_completed: name of the assembly step that has completed
        """
        assembly = robotic_arm_assembly.RoboticArmAssembly()
        step_completed, message = assembly.resume_assembly_from_last_step(step_completed)

        result = {
            "func_type": "roboticarm_process",
            "robot_name": robot_name,
            "step_completed": step_completed,
            "content": message 
        }
        
        return result    