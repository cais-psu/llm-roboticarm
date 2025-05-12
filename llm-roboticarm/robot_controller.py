# robot_controller.py

import urx
import socket
import time
import math

class RobotController:
    def __init__(self, config):
        self.config = config
        self.ip = config["ip"]
        self.dashboard_port = config["dashboard_port"]
        self.params = config["params"]
        self.gripper_programs = self.params.get("gripper_programs", {})
        self.robot = None

        try:
            self.robot = urx.Robot(self.ip)
            time.sleep(1)
            self.control_gripper("activate")
            print(f"[RobotController] Connected to robot at {self.ip}")
        except Exception as e:
            print(f"[RobotController] Failed to connect to robot: {e}. Running in offline mode.")

    def send_dashboard(self, cmd, wait=0.2):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.dashboard_port))
            s.sendall((cmd + "\n").encode())
            time.sleep(wait)
            return s.recv(1024).decode().strip()

    def run_program(self, filename, delay=1):
        self.send_dashboard(f"load {filename}")
        self.send_dashboard("play")
        time.sleep(delay)
        self.send_dashboard("stop")

    def control_gripper(self, action):
        if action not in self.gripper_programs:
            print(f"[ERROR] Gripper action '{action}' not defined.")
            return
        self.run_program(self.gripper_programs[action])

    def move_to_pose(self, coords_mm_deg):
        if not self.robot:
            print("[RobotController] Robot is not connected. Skipping move_to_pose.")
            return
        pos_m = [c / 1000.0 for c in coords_mm_deg[:3]]
        orient_rad = [math.radians(a) for a in coords_mm_deg[3:]]
        pose = pos_m + orient_rad
        self.robot.movel(pose, acc=self.params["acc"], vel=self.params["vel"])
        #time.sleep(0.1)

    def move_to_joint_pose(self, joint_angles_deg):
        """
        Move robot using joint space control (movej).

        Parameters:
            joint_angles_deg (list): Joint angles in degrees (length must match DOF).
        """
        joint_angles_rad = [math.radians(a) for a in joint_angles_deg]
        self.robot.movej(joint_angles_rad, acc=1, vel=1)
        time.sleep(0.1)

    def go_home(self):
        self.move_to_pose([313.33, -193.72, 487.86, -127.29 ,127.26,-0.00])

    def close(self):
        self.robot.close()
