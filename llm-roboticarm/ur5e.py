import urx
import time

def degrees_to_radians(joint_angles_deg):
    return [angle * 3.1416 / 180 for angle in joint_angles_deg]

# Replace with your UR5e's IP address
robot_ip = "192.168.1.172"

# Connect to the robot
robot = urx.Robot(robot_ip)

try:
    # Define target joint angles in degrees
    target_joints_deg = [180, -90, 90, 0, 90, 0]

    # Convert to radians before sending
    target_joints_rad = degrees_to_radians(target_joints_deg)

    print("Moving to joint position (degrees)...")
    robot.movej(target_joints_rad, acc=0.5, vel=0.2)
    time.sleep(2)

finally:
    robot.close()
    print("Connection closed.")
