import torch
from pathlib import Path

def check_model_path(path_str):
    path = Path(path_str)
    
    if path.exists() and path.is_file():
        try:
            # Attempt to load the model
            model = torch.hub.load('llm-roboticarm/ultralytics_yolov5_master', 'custom', path, source='local',force_reload=True)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(f"Path does not exist or is not a file: {path}")
        return False
# Replace with your actual model path
model_path = 'llm-roboticarm/vision_data/combined.pt'
check_model_path(model_path)