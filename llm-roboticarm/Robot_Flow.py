import json
import cv2
from pathlib import Path
from xarmlib.wrapper import XArmAPI
from inference_sdk import InferenceHTTPClient


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class XArm:
    def __init__(self, params_movement,
                 camera_index: int = 1,
                 save_folder: str = 'vision_data',
                 robot_ip: str = '192.168.1.240'):
        """
        Controls an XArm for picking and placing gears based on vision detections.

        :param params_movement: dict or JSON string with 'gear1' and 'gear2' steps
        :param camera_index: index for cv2.VideoCapture
        :param save_folder: where to save the captured frame
        :param robot_ip: IP address of the XArm
        """
        # initialize z direction
        self.z_list = [195, 185]
        self.z_count = 0

        # Parse movement parameters
        if isinstance(params_movement, str):
            self.params_movement = json.loads(params_movement)
        else:
            self.params_movement = params_movement
        self.gear_steps = {
            'Gear_Small': self.params_movement.get('gear3', []),
            'Gear_Big':   self.params_movement.get('gear4', [])
        }

        # Initialize robot
        self.arm = XArmAPI(robot_ip, baud_checkset=False)
        _, self.base_pos = self.arm.get_position()
        self.settings = {
            'speed':        280,
            'acc':          10000,
            'radius':       -1,
            'wait':         True,
            'grip_speed':   1000,
            'auto_enable':  True
        }
        # Move to initial detect position
        self.arm.set_position(250, -150, 445, 180, 0, 0,
                              speed=self.settings['speed'],
                              mvacc=self.settings['acc'],
                              radius=self.settings['radius'],
                              wait=self.settings['wait'])

        # Vision client and storage
        self.capture_index = camera_index
        self.save_folder = Path(save_folder)
        ensure_dir(self.save_folder)
        self.vision_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="SV6Y70SgNrmvw9mLNlcS"
        )

    def capture_frame(self) -> Path:
        """Capture a frame, save as JPEG, and return its path."""
        cap = cv2.VideoCapture(self.capture_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")

        frame = cv2.resize(frame, (640, 480))
        frame = frame[:435, :640]
        save_path = self.save_folder / 'image_capture.jpg'
        if not cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"Failed to write image to {save_path}")
        return save_path

    def process_frame(self, image_path: str):
        """Run detection workflow and return bounding boxes by class."""
        result = self.vision_client.run_workflow(
            workspace_name="gearassemblydetection",
            workflow_id="detect-count-and-visualize-3",
            images={"image": image_path},
            use_cache=True
        )
        # Extract boxes per class
        boxes = {}
        for image_rec in result:
            preds = image_rec['predictions']['predictions']
            for det in preds:
                cls = det['class']
                coords = (det['x'], det['y'], det['width'], det['height'])
                boxes.setdefault(cls, []).append(coords)
        return boxes

    def adjust_coordinates(self, boxes: dict) -> dict:
        """Convert vision coords to robot world coords."""
        adjustments = {}
        for cls, pts in boxes.items():
            x_px, y_px, *_ = pts[0]  # first detection
            # map pixel to real-world with linear transform
            x_rw = self.base_pos[0] - 0.6 * y_px + 215
            y_rw = self.base_pos[1] - 0.6 * x_px + 215
            adjustments[cls] = (x_rw, y_rw)
        return adjustments

    def move_gears(self, adjusted: dict):
        """Execute pick-and-place steps for each gear type."""

        for cls, steps in self.gear_steps.items():
            origin = adjusted.get(cls)
            if origin is None:
                continue
            x0, y0 = origin
            for step in steps:
                if isinstance(step, int):  # gripper
                    self.arm.set_gripper_position(
                        step,
                        wait=self.settings['wait'],
                        speed=self.settings['grip_speed'],
                        auto_enable=self.settings['auto_enable']
                    )
                elif len(step) == 4:  # offset from center
                    self.arm.set_position(
                        x0 , y0 ,*step,
                        speed=self.settings['speed'],
                        mvacc=self.settings['acc'],
                        radius=self.settings['radius'],
                        wait=self.settings['wait']
                    )
                elif len(step) == 3:  # place gear on the base
                    x,y = adjusted.get('Base')
                    z = self.z_list[self.z_count % len(self.z_list)]
                    self.z_count += 1
                    offsets = {
                        'small gear': (-7.5, -7.5),
                        'big gear': (12.5, 12.5),
                    }
                    dx,dy = offsets.get(cls.lower(),(0,0))
                    x1, y1 = x+dx, y+dy
                    self.arm.set_position(
                    x1, y1 ,z,
                    speed=self.settings['speed'],
                    mvacc=self.settings['acc'],
                    radius=self.settings['radius'],
                    wait=self.settings['wait']
                        )
                else:  # absolute coords
                    self.arm.set_position(
                        *step,
                        speed=self.settings['speed'],
                        mvacc=self.settings['acc'],
                        radius=self.settings['radius'],
                        wait=self.settings['wait']
                    )

    def robotic_assembly(self):
        """Full pipeline: capture → detect → adjust → move."""
        img_path = str(self.capture_frame())
        boxes = self.process_frame(img_path)
        adjusted = self.adjust_coordinates(boxes)
        self.move_gears(adjusted)


if __name__ == '__main__':
    params_path = 'vision_data/Train/Gear.json'
    with open(params_path,'r') as f:
        params = json.load(f)
    arm = XArm(params)
    arm.robotic_assembly()