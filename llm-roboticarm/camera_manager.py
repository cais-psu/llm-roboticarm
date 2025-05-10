# Updated camera_manager.py based on process_camera style
import json
import cv2
import torch
import pathlib
import threading
import numpy as np
from collections import deque
import time
import datetime
import os
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

        for cam_type in self.camera_config.keys():
            window_name = f"CameraManager - {cam_type}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            cv2.moveWindow(window_name, 100, 100)     # Place window on screen

            if cam_type in processed_frames:
                frame = processed_frames[cam_type]
                cv2.imshow(window_name, frame)
            else:
                w, h = self.camera_config[cam_type]["frame_size"]
                empty_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(empty_frame, f"No frame from {cam_type}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, empty_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            return True

    def capture_scene_with_detections(self, cam_type: str = "cameraA") -> np.ndarray:
        """
        Captures a fresh frame from the specified camera, runs YOLO detection, overlays bounding boxes,
        and returns the annotated image (no saving).

        :param cam_type: The camera identifier as specified in the config.
        :return: Annotated frame (np.ndarray) or None if unsuccessful.
        """
        if cam_type not in self.captures or cam_type not in self.models:
            print(f"[CameraManager] Camera or model not initialized for: {cam_type}")
            return None

        cap = self.captures[cam_type]

        # Flush the buffer and grab a fresh frame
        cap.read()
        time.sleep(0.1)  # Small delay to ensure new frame is captured
        ret, frame = cap.read()
        if not ret:
            print(f"[CameraManager] Could not read frame from camera '{cam_type}'.")
            return None

        cam_cfg = self.camera_config[cam_type]
        frame = cv2.resize(frame, tuple(cam_cfg['frame_size']))
        frame = frame[cam_cfg['crop'][0]:cam_cfg['crop'][1], cam_cfg['crop'][2]:cam_cfg['crop'][3]]
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.models[cam_type](input_frame)
        coords_plus = results.pandas().xyxy[0]

        for _, row in coords_plus.iterrows():
            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame


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

if __name__ == "__main__":
    with open("llm-roboticarm/initialization/resources/sensors/camera.json") as f:
        camera_config = json.load(f)

    cm = CameraManager(camera_config)
    cm.capture_scene_with_detections(cam_type="cameraA")
