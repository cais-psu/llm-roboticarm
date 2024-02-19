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

    def sorting(self, product_name: str) -> dict:
        """
        This is a sorting function. When executed the product is sorted.
        The function returns whether the operation was successful.

        :param product_name: name of the product to be sorted
        """
        print(f"Sorting product {product_name}")
        
        result = {
            "func_type": "roboticarm_process",
            "product_name": product_name,
            "status": "completed",
            "content": f"Sorting {product_name} is successfully completed." 
        }
        return result
    
    def assembly(self, product_name: str) -> dict:
        """
        This is an assembly function. When executed the product is assemblied.
        The function returns whether the operation was successful.

        :param product_name: name of the product to be assemblied
        """
        assembly = robotic_arm_assembly.RoboticArmAssembly()
        assembly.robotic_assembly()

        result = {
            "func_type": "roboticarm_process",
            "product_name": product_name,
            "status": "completed",
            "content": f"Assembly of {product_name} is successfully completed." 
        }
        return result