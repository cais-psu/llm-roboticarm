import os
import json
from xarmlib.wrapper import XArmAPI
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import cv2
from inference_sdk import InferenceHTTPClient


class XArm:
    def __init__(self, params_movement):
        """
        Initializes the XArm, XArm is used to pick and place gear 1 on the base

        Parameters
        ----------
        params_movement
            JSON-formatted string or dictionary with general configuration parameters.
        """

        # Initial setting of XArm
        self.arm = XArmAPI('192.168.1.240', baud_checkset=False)  # IP address of the robot
        self.a, self.base = self.arm.get_position()
        self.params_settings = {
            'grip_speed': 1000,
            'radius': -1,
            'auto_enable': True,
            'wait': True,
            'speed': 280,
            'acc': 10000,
            'angle_speed': 20,
            'angle_acc': 500,
            'events': {},
            'callback_in_thread': True,
            'quit': False,
            'default_angles': [180, 0, 0]
        }

        # Initial Position for detecting objects
        self.arm.set_position(*[250, -150, 445, 180.0, 0.0, 0.0], speed=self.params_settings['speed'],
                              mvacc=self.params_settings['acc'], radius=self.params_settings['radius'],
                              wait=self.params_settings['wait'])

        # Initialize attributes to track the last logged line and timestamp
        self.last_logged_line = None
        self.last_logged_func = None
        self.inside_loop = False

        # Load the movement of the robot to pick up gear 1
        if isinstance(params_movement, str):
            self.params_movement = json.loads(params_movement)
        else:
            self.params_movement = params_movement
        self.gear1 = params_movement.get('gear1', [])
        self.gear2 = params_movement.get('gear2', [])


    def _coordinates_adjustment(self,gear_small_boxes,gear_big_boxes,base_boxes):
        """
        adjust the coordinates of target object so that robot could pick.

        Parameters
        ----------
        coord_list :
            List of coordinates representing the bounding box for the detected object.
        Returns
        -------
        str
            A message indicating the status of the assembly step.
        """

        # for small gear
        x_gs = self.base[0] + gear_small_boxes[0][1]*-0.6 + 205
        y_gs = self.base[1] + gear_small_boxes[0][0]*-0.6 + 215
        new_small_gear = [x_gs, y_gs]

        # for big gear
        x_gb = self.base[0] + gear_big_boxes[0][1]*-0.6 + 215
        y_gb = self.base[1] + gear_big_boxes[0][0]*-0.6 + 215
        new_big_gear = [x_gb, y_gb]

        # for base
        x_b = self.base[0] + base_boxes[0][1]*-0.6 + 215
        y_b = self.base[1] + base_boxes[0][0]*-0.6 + 215
        new_base = [x_b, y_b]

        return new_small_gear, new_big_gear, new_base



    def process_frame(self,frame_capture):

        client = InferenceHTTPClient(

            api_url="https://serverless.roboflow.com",
            api_key="SV6Y70SgNrmvw9mLNlcS"
        )

        result = client.run_workflow(
            workspace_name="gearassemblydetection",
            workflow_id="detect-count-and-visualize-3",
            images={
                "image": frame_capture
            },
            use_cache=True  # cache workflow definition for 15 minutes
        )

        # 1) Get the position of small gear
        all_gear_small = [
            det
            for image_rec in result
            for det in image_rec['predictions']['predictions']
            if det['class'] == 'Gear_Small'
        ]

        gear_small_boxes = [
            (d['x'], d['y'], d['width'], d['height'])
            for d in all_gear_small
        ]

        # 2) Get the position of big gear
        all_gear_big = [
            det
            for image_rec in result
            for det in image_rec['predictions']['predictions']
            if det['class'] == 'Gear_Big'
        ]

        gear_big_boxes = [
            (d['x'], d['y'], d['width'], d['height'])
            for d in all_gear_big
        ]

        # 3) Get boundary position for base
        all_base = [
            det
            for image_rec in result
            for det in image_rec['predictions']['predictions']
            if det['class'] == 'Base'
        ]

        base_boxes = [
            (d['x'], d['y'], d['width'], d['height'])
            for d in all_base
        ]

        # 4) Get the image size
        #image_size = result[0]['predictions']['image']

        return gear_small_boxes, gear_big_boxes, base_boxes

    def _movement(self,new_small_boxes, new_gear_big_boxes, new_base_boxes):
        """


        # Check for any arm errors or quit signals
        if self.arm.error_code != 0 or self.params_settings['quit']:
            print("Error or quit condition encountered.")
            return

        # Execute each step in the selected set of instructions
        """

        for step in self.gear1:
            if isinstance(step, int):
                # Set gripper position if the step is an integer
                self.arm.set_gripper_position(
                    step,
                    wait=self.params_settings['wait'],
                    speed=self.params_settings['grip_speed'],
                    auto_enable=self.params_settings['auto_enable']
                )
            elif len(step) == 4:
                # Move the arm using the combined object coordinates and step details
                self.arm.set_position(
                    new_small_boxes[0], new_small_boxes[1], *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )
            elif len(step) == 3:
                self.arm.set_position(
                    new_base_boxes[0]-7.5, new_base_boxes[1], *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )
            else:
                # Move the arm using step coordinates as provided
                self.arm.set_position(
                    *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )

        for step in self.gear2:
            if isinstance(step, int):
                # Set gripper position if the step is an integer
                self.arm.set_gripper_position(
                    step,
                    wait=self.params_settings['wait'],
                    speed=self.params_settings['grip_speed'],
                    auto_enable=self.params_settings['auto_enable']
                )
            elif len(step) == 4:
                # Move the arm using the combined object coordinates and step details
                self.arm.set_position(
                    new_gear_big_boxes[0], new_gear_big_boxes[1], *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )
            elif len(step) == 3:
                # Move the arm using the combined object coordinates and step details
                self.arm.set_position(
                    new_gear_big_boxes[0], new_gear_big_boxes[1], *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )
            else:
                # Move the arm using step coordinates as provided
                self.arm.set_position(
                    *step,
                    speed=self.params_settings['speed'],
                    mvacc=self.params_settings['acc'],
                    radius=self.params_settings['radius'],
                    wait=self.params_settings['wait']
                )

    def robotic_assembly(self):
        """
        Starts the robotic to pick and place the gear 1
        """

        # Initialize the camera
        capture = cv2.VideoCapture(1)

        object_counts, object_coords = {}, {}

        # Capture a single frame from the camera
        ret, frame = capture.read()
        frame = cv2.resize(frame, (640, 480))
        frame=frame[0:435,0:640]
        if not ret:
            capture.release()
            cv2.destroyAllWindows()
            return "error", "Failed to capture frame from the camera."


        folder = "vision_data"
        save_path = os.path.join(folder, "image_capture.jpg")

        # convert the frame to jpg
        cv2.imwrite(save_path, frame,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

        path = 'vision_data/image_capture.jpg'

        # Process the frame with RobotFlow, localize the positions of target object
        gear_small_boxes, gear_big_boxes, base_boxes = self.process_frame(path)

        new_small_boxes, new_gear_big_boxes, new_base_boxes = self._coordinates_adjustment(gear_small_boxes, gear_big_boxes, base_boxes)

        self._movement(new_small_boxes, new_gear_big_boxes, new_base_boxes)

if __name__ == "__main__":
    params_movement_path = 'Gear.json'
    with open(params_movement_path, 'r') as file:
        params_movement = json.load(file)
    assembly = XArm(params_movement)
    assembly.robotic_assembly()