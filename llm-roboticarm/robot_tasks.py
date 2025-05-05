import time
import json
import threading
import numpy as np
import cv2
from robot_controller import RobotController
from camera_manager import CameraManager

class RobotTask:
    def __init__(self, robot_controller, camera_manager, robot_config, product_config):
        self.robot_controller = robot_controller
        self.camera_manager = camera_manager
        self.robot_config = robot_config
        self.product_config = product_config

        self.assembly_steps = self.product_config["assembly_steps"]
        self.dropoff_positions = self.product_config["dropoff_positions"]

        self.camera_name = self.robot_config["camera_types"][0]
        self.z_pick = self.robot_config["z_pick"]
        self.z_above = self.robot_config["z_above"]
        self.pickup_orientation = self.robot_config["pickup_orientation"]
        self.intermediate_joint_pose = self.robot_config["intermediate_joint_pose"]
        self.place_orientations = self.robot_config["place_orientations"]
        self.grasp_strategies = self.robot_config["grasp_strategies"]
        self.gripper_programs = self.robot_config["params"]["gripper_programs"]

        # Calibration
        calibration = self.robot_config.get("camera_calibration", {})
        camera_points = np.array(calibration.get("camera_points", []), dtype=np.float32)
        robot_points = np.array(calibration.get("robot_points", []), dtype=np.float32)

        if len(camera_points) >= 3 and len(robot_points) >= 3:
            self.affine_matrix, _ = cv2.estimateAffine2D(camera_points, robot_points)
        else:
            raise ValueError("Insufficient calibration data.")

        self.robot_controller.go_home()

    def pixel_to_robot(self, cx, cy):
        pt = np.dot(self.affine_matrix, np.array([cx, cy, 1]))
        return pt[0], pt[1]

    def apply_grasp_strategy(self, strategy, x1, y1, x2, y2):
        if strategy == "tip-right":
            return x2, int((y1 + y2) / 2)
        elif strategy == "tip-left":
            return x1, int((y1 + y2) / 2)
        elif strategy == "bottom":
            return int((x1 + x2) / 2), y2
        elif strategy == "top":
            return int((x1 + x2) / 2), y1
        elif strategy == "above-top":
            return int((x1 + x2) / 2), y1 - 5
        else:
            return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def detect_object(self, component):
        self.camera_manager.process_all_cameras()
        coords_dict = self.camera_manager.object_coords_dict.get(self.camera_name, {})
        bbox_dict = self.camera_manager.object_bboxes_dict.get(self.camera_name, {})
        matching_keys = [key for key in coords_dict if key.startswith(component)]
        if not matching_keys:
            return None, None, None
        key = matching_keys[0]
        if key not in bbox_dict:
            return key, None, None
        x1, y1, x2, y2 = bbox_dict[key]
        strategy = self.grasp_strategies.get(component, {}).get("strategy", "center")
        cx, cy = self.apply_grasp_strategy(strategy, x1, y1, x2, y2)
        return key, cx, cy

    def pick(self, x, y, component):
        gripper_action = self.grasp_strategies.get(component, {}).get("gripper_action", "close")
        pick_pos = [x, y, self.z_pick]
        pick_above = [x, y, self.z_above]
        self.robot_controller.move_to_pose(pick_above + self.pickup_orientation)
        self.robot_controller.move_to_pose(pick_pos + self.pickup_orientation)
        self.robot_controller.control_gripper(gripper_action)
        self.robot_controller.move_to_pose(pick_above + self.pickup_orientation)

    def intermediate(self):
        self.robot_controller.move_to_joint_pose(self.intermediate_joint_pose)

    def place(self, component):
        place_position = self.dropoff_positions.get(component)
        if not place_position:
            print(f"[TASKS] No dropoff defined for '{component}'")
            return False
        place_above = place_position[:2] + [self.z_above]
        orientation = self.place_orientations.get(component, self.pickup_orientation)
        self.robot_controller.move_to_pose(place_above + orientation)
        self.robot_controller.move_to_pose(place_position + orientation)
        self.robot_controller.control_gripper("open")
        self.robot_controller.move_to_pose(place_above + orientation)
        return True

    def _perform_task(self, component):
        print(f"[TASKS] Waiting to detect '{component}' for assembly...")
        while True:
            time.sleep(0.3)
            key, cx, cy = self.detect_object(component)
            if key and cx is not None:
                print(f"[TASKS] Detected '{key}' at ({cx}, {cy})")
                x_pick, y_pick = self.pixel_to_robot(cx, cy)
                self.pick(x_pick, y_pick, component)
                self.intermediate()
                if self.place(component):
                    print(f"[TASKS] Assembly of '{component}' completed.")
                    self.robot_controller.go_home()
                    return True
            print(f"[TASKS] '{component}' not detected. Waiting...")
            time.sleep(0.5)

    def assembly_continuous(self):
        def task_loop():
            while True:
                for component in self.assembly_steps:
                    self._perform_task(component)
        threading.Thread(target=task_loop, daemon=True).start()
        while True:
            self.camera_manager.process_all_cameras()
            time.sleep(0.05)

    def assembly(self):
        for component in self.assembly_steps:
            success = self._perform_task(component)
            if not success:
                print(f"[TASKS] Failed to complete task for '{component}'")

    def refresh_config(self):
        pass
    
if __name__ == "__main__":
    with open("llm-roboticarm/initialization/resources/robots/robots.json") as f:
        robot_config = json.load(f)["ur5e"]
    with open("llm-roboticarm/initialization/resources/sensors/camera.json") as f:
        camera_config = json.load(f)
    with open("llm-roboticarm/initialization/products/products.json") as f:
        product_config = json.load(f)

    robot_controller = RobotController(robot_config)
    camera_manager = CameraManager(camera_config)
    robot_task = RobotTask(robot_controller, camera_manager, robot_config, product_config)
    robot_task.assembly()

    #robot_task.assembly_continuous()
