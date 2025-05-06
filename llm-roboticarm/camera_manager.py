# Updated camera_manager.py based on process_camera style
import json
import cv2
import torch
import pathlib
import threading
import numpy as np
from collections import deque
import time

pathlib.PosixPath = pathlib.WindowsPath

class CameraManager:
    def __init__(self, config_path):
        self.models = {}
        self.captures = {}
        self.frames = {}
        self.object_counts_dict = {}
        self.object_coords_dict = {}
        self.object_bboxes_dict = {}
        self.camera_config = config_path

        for name, cfg in self.camera_config.items():
            index = cfg["camera_index"]
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"[CameraManager] Camera '{name}' not available.")
            self.captures[name] = cap

            model_path = cfg.get("model_path")
            if model_path:
                self.models[name] = torch.hub.load(
                    'llm-roboticarm/ultralytics_yolov5_master',
                    'custom',
                    path=model_path,
                    source='local',
                    force_reload=True
                )

    def process_all_cameras(self):
        processed_frames = {}
        camera_suffix_map = {
            cam: str(idx + 1)
            for idx, cam in enumerate(self.camera_config.keys())
        }

        def process_camera(cam_type):
            if cam_type not in self.captures or cam_type not in self.models:
                print(f"[CameraManager] Camera or model not initialized for: {cam_type}")
                return

            cap = self.captures[cam_type]
            ret, frame = cap.read()
            if not ret:
                print(f"[CameraManager] Failed to read from camera: {cam_type}")
                return

            cam_cfg = self.camera_config[cam_type]
            frame = cv2.resize(frame, tuple(cam_cfg['frame_size']))
            frame = frame[cam_cfg['crop'][0]:cam_cfg['crop'][1], cam_cfg['crop'][2]:cam_cfg['crop'][3]]
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.models[cam_type](input_frame)
            coords_plus = results.pandas().xyxy[0]

            object_counts = {}
            object_coords = {}
            object_bboxes = {}
            centroids = []

            for _, row in coords_plus.iterrows():
                raw_name = row['name']
                base_name = raw_name.split('-')[0]

                suffix = camera_suffix_map.get(cam_type, '')
                name = f"{base_name}-{suffix}" if suffix else base_name

                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if any(np.linalg.norm(np.array([cx, cy]) - np.array(existing)) < 15 for existing in centroids):
                    continue
                centroids.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                object_counts[name] = object_counts.get(name, 0) + 1
                object_coords[name] = [cx, cy]
                object_bboxes[name] = [x1, y1, x2, y2]

            processed_frames[cam_type] = frame
            self.object_counts_dict[cam_type] = object_counts
            self.object_coords_dict[cam_type] = object_coords
            self.object_bboxes_dict[cam_type] = object_bboxes


        threads = []
        for cam_type in self.camera_config.keys():
            t = threading.Thread(target=process_camera, args=(cam_type,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for cam_type, frame in processed_frames.items():
            cv2.imshow(f"CameraManager - {cam_type}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            return True

'''
Bounding Box:
| Term                | Meaning                                                                | Example                                                   |
| ------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------- |
| **`object_coords`** | The **center (cx, cy)** of the bounding box (for pick/grasp location). | `(320, 240)` – center pixel of the object                 |
| **`object_bboxes`** | The **full bounding box** of the object: `(x1, y1, x2, y2)`            | `(280, 200, 360, 280)` – top-left to bottom-right corners |

(x1, y1) ------------------
|                         |
|         (cx, cy)        |
|                         |
------------------ (x2, y2)
'''

# Optional standalone test
if __name__ == "__main__":
    cm = CameraManager()
    while True:
        cm.process_all_cameras()
