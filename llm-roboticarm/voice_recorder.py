import wave
import struct
from pvrecorder import PvRecorder
import time, os

class VoiceRecorder:
    def __init__(self, device_index=0, frame_length=512):
        self.device_index = device_index
        self.frame_length = frame_length
        self.recording = False
        self.audio = []

    def record_audio(self):
        recorder = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
        try:
            recorder.start()
            print("Recording started!!")
            self.audio = []
            while self.recording:
                frame = recorder.read()
                self.audio.extend(frame)
        finally:
            recorder.stop()
            recorder.delete()
            print("Recording stopped.")
            self.save_recording()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            return True
        else:
            print("Recording is already in progress.")
            return False

    def stop_recording(self):
        if self.recording:
            self.recording = False
            return True
        else:
            print("Recording is not active.")
            return False

    def save_recording(self):
        if not self.audio:
            print("No audio recorded.")
            return
        path = 'voice_command.wav'
        #backup_path = 'voice_command_backup_{}.wav'.format(time.strftime("%Y%m%d-%H%M%S"))

        # Check if the file already exists
        if os.path.exists(path):
            # Rename the existing file by moving it to a new location
            os.remove(path)
            #os.rename(path, backup_path)
            #print(f"Existing file renamed to {backup_path}")

        # Now save the new recording
        with wave.open(path, 'w') as f:
            f.setparams((1, 2, 16000, 0, "NONE", "NONE"))
            f.writeframes(struct.pack('<' + ('h' * len(self.audio)), *self.audio))
        print("Recording saved successfully.")