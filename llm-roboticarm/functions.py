import utils
import sys, os
import time
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
            config_data = utils.load_json_data(init)
            all_configs.update(config_data)
        return all_configs

    def sorting(self, robot_name: str) -> dict:
        """
        This is a sorting function. When executed the product is sorted.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that sorts the product
        """
        print(f"Sorting product {robot_name}")
        time.sleep(30)
        
        result = {
            "func_type": "sorting_process",
            "robot_name": robot_name,
            "step_already_done": "completed",
            "content": f"Sorting by {robot_name} is successfully completed." 
        }
        return result
    
    def assembly(self, robot_name: str) -> dict:
        """
        This is an assembly function. When executed, the product is assemblied.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that assembles the product
        """
        
        #when works
        assembly = robotic_arm_assembly.RoboticArmAssembly()
        step_already_done, message = assembly.start_robotic_assembly()
        
        #step_already_done, message = "completed", "All steps for the assembly are successfully completed."
        
        result = {
            "func_type": "assembly_process",
            "robot_name": robot_name,
            "step_already_done": step_already_done,
            "content": message 
        }
            
        return result   
    
    def resume_assembly(self, robot_name: str, step_already_done: str) -> dict:
        """
        This is a resuming assembly function after the error resolve is finished, done or completed. When executed, the product will do assembly from the point where it left off.
        The function returns whether the operation was successful.

        :param robot_name: name of the robot that resumes the assembly of the product
        :param step_already_done: name of the assembly step that has been already completed
        """
        
        #when works
        assembly = robotic_arm_assembly.RoboticArmAssembly()
        step_already_done, message = assembly.resume_assembly_from_last_step(step_already_done)

        #step_already_done, message = "completed", "All steps for the assembly are successfully completed."

        result = {
            "func_type": "roboticarm_process",
            "robot_name": robot_name,
            "step_already_done": step_already_done,
            "content": message 
        }
        
        return result