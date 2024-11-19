import os
import threading
import sys
import json
import cv2
import torch

from rag_handler import RAGHandler
from prompts import VERBAL_UPDATES_INSTRUCTIONS

from xarmlib.wrapper import XArmAPI
from voice_control import VoiceControl
import functools

import pathlib
pathlib.PosixPath=pathlib.WindowsPath

import logging
import log_setup

import torch
import cv2

class RoboticArmAssembly:
    def __init__(self, params_general, params_movement):
        """
        Initializes the RoboticArmAssembly class, setting up the logger, 
        robotic arm API, parameters, and voice control.
        
        Parameters
        ----------
        params_general
            JSON-formatted string or dictionary with general configuration parameters.
        """
        # Set up agent-specific logger
        self.log_setup = log_setup.LogSetup(name="xArm")
        self.log_setup.setup_logging("process")
        self.logger = self.log_setup.logger_process

        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.sop_handler = RAGHandler('llm-roboticarm/initialization/robots/specification/xArm_SOP.pdf', 'pdf', self.openai_api_key)

        self.arm = XArmAPI('192.168.1.240', baud_checkset=False)
        self.variables = {}
        self.a,self.base=self.arm.get_position()

        self.adaptation = False
        
        self.params_settings = {
            'grip_speed':800,
            'radius':-1,
            'auto_enable':True,
            'wait':True,
            'speed': 180,
            'acc': 10000,
            'angle_speed': 20,
            'angle_acc': 500,
            'events': {},
            'variables': self.variables,
            'callback_in_thread': True,
            'quit': False,
            'default_angles':[180,0,0]
        }

        # Initial Position for detecting objects
        self.arm.set_position(*[250,-150,445,180.0,0.0,0.0], speed=self.params_settings['speed'], mvacc=self.params_settings['acc'], radius=self.params_settings['radius'], wait=self.params_settings['wait'])

        # Load the parameters from JSON if provided as a string
        self.step_working_on = None
        self.voice_control = VoiceControl()

        # Initialize attributes to track the last logged line and timestamp
        self.last_logged_line = None
        self.last_logged_func = None
        self.inside_loop = False  

        if isinstance(params_general, str):
            self.params_general = json.loads(params_general)
        else:
            self.params_general = params_general

        if isinstance(params_movement, str):
            self.params_movement = json.loads(params_movement)
        else:
            self.params_movement = params_movement

        self.housing_set = params_movement.get('housing_set', [])
        self.wedge_set = params_movement.get('wedge_set', [])
        self.spring_set = params_movement.get('spring_set', [])
        self.cap_set = params_movement.get('cap_set', [])
        self.housing_90 = params_movement.get('housing_90', [])
        self.wedge_90 = params_movement.get('wedge_90', [])
        self.spring_90 = params_movement.get('spring_90', [])
        self.cap_90 = params_movement.get('cap_90', [])

    def trace_lines(self, frame, event, arg):
        """
        Traces line-by-line execution of functions to provide detailed logging.
        
        Parameters
        ----------
        frame : Frame
            Current frame in the code being executed.
        event : str
            Type of event (e.g., 'line').
        arg : str
            Arguments for the event.
        """        
        if event == 'line':
            code = frame.f_code
            lineno = frame.f_lineno
            filename = code.co_filename
            name = code.co_name

            # Ensure we only log lines from the current file
            if filename != __file__:
                return self.trace_lines

            # Detect if we're entering a loop by checking the function name
            if name == self.last_logged_func:
                self.inside_loop = True
            else:
                self.inside_loop = False

            # Log the line execution if it is not inside a loop
            if not self.inside_loop:
                self.logger.info(f'Executing line {lineno} in {name} of {filename}')
                self.last_logged_line = lineno
                self.last_logged_func = name

        return self.trace_lines
    
    @staticmethod
    def log_execution(func):
        """
        Decorator to log the start and end of function executions.
        """        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Ensure last_logged_line and last_logged_func are initialized for the instance
            if not hasattr(self, 'last_logged_line'):
                self.last_logged_line = None
            if not hasattr(self, 'last_logged_func'):
                self.last_logged_func = None
            if not hasattr(self, 'inside_loop'):
                self.inside_loop = False

            sys.settrace(lambda frame, event, arg: self.trace_lines(frame, event, arg))
            try:
                self.logger.info(f"Running {func.__name__}...")
                result = func(self, *args, **kwargs)
                self.logger.info(f"Completed {func.__name__}.")
            finally:
                sys.settrace(None)  # Ensure the trace is disabled after function execution
            return result
        return wrapper
    
    def movement(self, bounding_box_coords, object_set, object_90, pixel_x_coord, pixel_y_coord):
        """
        Moves the robotic arm to detected objects for gripping and assembly.

        Parameters
        ----------
        bounding_box_coords : list
            Coordinates of the bounding box for the detected object.
        object_set : list
            Assembly instructions for vertical orientation.
        object_90 : list
            Assembly instructions for horizontal orientation.
        pixel_x_coord : int
            X-coordinate for object positioning.
        pixel_y_coord : int
            Y-coordinate for object positioning.
        """
        # Determine orientation and select appropriate set of instructions
        is_vertical = (bounding_box_coords[2] - bounding_box_coords[0]) <= (bounding_box_coords[3] - bounding_box_coords[1])
        selected_set = object_set if is_vertical else object_90

        # Check for any arm errors or quit signals
        if self.arm.error_code != 0 or self.params_settings['quit']:
            print("Error or quit condition encountered.")
            return

        # Execute each step in the selected set of instructions
        for step in selected_set:
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
                    pixel_x_coord, pixel_y_coord, *step,
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

    def find_available_cameras(self):
        """
        Attempts to open cameras to check which indices are available.
        
        Returns
        -------
        list
            List of available camera indices.
        """

        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def _verbal_updates(self, step_working_on: str):
        """
        Provides verbal updates based on the assembly step.

        Parameters
        ----------
        step_working_on : str
            Name of the current assembly step.
        """
        message = self.sop_handler.retrieve(f"Assembly step working on: {step_working_on}." + VERBAL_UPDATES_INSTRUCTIONS)
        threading.Thread(target=self.voice_control.text_to_speech, args=(message, 0)).start()

    def _initialize_camera_and_model(self):
        """
        Initializes the object detection model and camera.

        Returns
        -------
        tuple
            A tuple containing the loaded model and initialized camera object.
        """
        path = 'llm-roboticarm/vision_data/combined.pt'
        model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local', force_reload=True)
        capture = cv2.VideoCapture(2)
        return model, capture

    def _process_and_display_frame(self, frame, model, object_counts, object_coords):
        """
        Processes a camera frame, performs object detection, and updates and displays object data.

        Parameters
        ----------
        frame : np.ndarray
            The camera frame to process.
        model : torch model
            The loaded object detection model.
        object_counts : dict
            A dictionary to store counts of detected objects.
        object_coords : dict
            A dictionary to store coordinates of detected objects.
        """
        frame = cv2.resize(frame, (640, 480))
        frame=frame[0:435,0:640]
        results = model(frame)
        coords_plus = results.pandas().xyxy[0]

        object_counts.clear()
        object_coords.clear()

        for _, row in coords_plus.iterrows():
            name = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            object_counts[name] = object_counts.get(name, 0) + 1
            object_coords[name] = [x1, y1, x2, y2]

        # Display object counts on frame
        y_offset = 30
        for obj_name, count in object_counts.items():
            cv2.putText(frame, f'{obj_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20
        cv2.imshow("Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            return True
        return False

    #@log_execution
    def _perform_specific_assembly_step(self, coord_list, step_type):
        """
        Performs an assembly step for the specified step type.

        Parameters
        ----------
        coord_list : list
            List of coordinates representing the bounding box for the detected object.
        step_type : str
            The type of assembly step to perform (e.g., "housing", "wedge", "spring", "cap").
        Returns
        -------
        str
            A message indicating the status of the assembly step.
        """

        # Mapping of step type to movement sets and offsets
        step_data = {
            "housing": {"set": self.housing_set, "set_90": self.housing_90, "x_offset": 235, "y_offset": 215},
            "wedge": {"set": self.wedge_set, "set_90": self.wedge_90, "x_offset": 245, "y_offset": 230},
            "spring": {"set": self.spring_set, "set_90": self.spring_90, "x_offset": 235, "y_offset": 230},
            "cap": {"set": self.cap_set, "set_90": self.cap_90, "x_offset": 247, "y_offset": 230},
        }

        # Get the appropriate movement sets and offsets based on the step type
        if step_type not in step_data:
            return f"Error: Invalid step type '{step_type}'."

        movement_set = step_data[step_type]["set"]
        movement_set_90 = step_data[step_type]["set_90"]
        x_offset = step_data[step_type]["x_offset"]
        y_offset = step_data[step_type]["y_offset"]

        # Calculate the midpoint coordinates based on the bounding box
        try:
            x_mid = (coord_list[0] + coord_list[2]) / 2
            y_mid = (coord_list[1] + coord_list[3]) / 2

            # Adjust coordinates to the robot’s base position
            x_adjusted = self.base[0] + y_mid * -0.6 + x_offset
            y_adjusted = self.base[1] + x_mid * -0.6 + y_offset

        except Exception as e:
            return f"Error: the {step_type} object was not detected: {str(e)}"

        # Perform the movement based on the detected object's orientation
        try:
            print("movement")
            #self.movement(coord_list, movement_set, movement_set_90, x_adjusted, y_adjusted)
        except Exception as e:
            return f"Error during {step_type} movement: {str(e)}"

        return f"{step_type.capitalize()} step completed successfully."

    def _execute_assembly_steps(self, assembly_steps, step_working_on, object_coords, cap):
        """
        Executes assembly steps based on detected objects and current working step.

        Parameters
        ----------
        assembly_steps : list
            List of assembly steps to perform.
        step_working_on : str
            The current step being worked on.
        object_coords : dict
            Dictionary with coordinates for each detected object.
        cap : cv2.VideoCapture
            The camera capture object to release in case of errors.

        Returns
        -------
        tuple
            A tuple containing the status ("continue" or step name) and a message.
        """
        if step_working_on not in assembly_steps:
            self.logger.info("Starting the robotic assembly process from the beginning")
            step_working_on = assembly_steps[0]

        current_index = assembly_steps.index(step_working_on)
        for step in assembly_steps[current_index:]:
            if step in object_coords:
                coords = object_coords[step]
                self.step_working_on = step
                message = self._perform_specific_assembly_step(coords, step)
                if 'error' in message.lower():
                    cap.release()
                    cv2.destroyAllWindows()
                    return step, message
            else:
                cap.release()
                cv2.destroyAllWindows()
                return step, f"Error: the {step} object was not detected"

        return "completed", ""

    #@log_execution
    def robotic_assembly(self, step_working_on: str):
        """
        Starts the robotic assembly process by performing each step sequentially.
        If the step is not part of the assembly steps, it starts from the beginning.

        :param step_working_on: The exact name of the assembly step currently being worked on. This should be used directly and never altered based on any other context or conditions.
        """

        # Start verbal updates asynchronously
        threading.Thread(target=self._verbal_updates, args=(step_working_on,)).start()

        # Initialize the model and camera
        model, capture = self._initialize_camera_and_model()
        object_counts, object_coords = {}, {}

        # Capture a single frame from the camera
        ret, frame = capture.read()
        if not ret:
            capture.release()
            cv2.destroyAllWindows()
            return "error", "Failed to capture frame from the camera."

        # Process frame and update object data
        self._process_and_display_frame(frame, model, object_counts, object_coords)

        # Retrieve assembly steps based on adaptation mode
        assembly_steps = self.params_general.get("assembly_steps", []) if self.adaptation else ["housing", "wedge", "spring", "cap"]

        # Execute the assembly steps based on detected objects
        result, message = self._execute_assembly_steps(assembly_steps, step_working_on, object_coords, capture)

        # Release resources after processing
        capture.release()
        cv2.destroyAllWindows()

        # Return the final status and message
        if result != "completed":
            return result, message  
        else:
            return "completed", "Assembly process completed successfully."

if __name__ == "__main__":
    params_general_path = 'llm-roboticarm/initialization/robots/specification/params_general.json'
    params_movement_path = 'llm-roboticarm/initialization/robots/specification/params_movement.json'

    with open(params_general_path, 'r') as file:
        params_general = json.load(file)

    with open(params_movement_path, 'r') as file:
        params_movement = json.load(file)

    assembly = RoboticArmAssembly(params_general, params_movement)
    #assembly.find_available_cameras()
    assembly.robotic_assembly(step_working_on="None")

    