import tkinter as tk
from tkinter import scrolledtext, filedialog
import os

class UserInterface:
    _instance = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("xArm User Interface")
        self.root.geometry("500x400")  # Adjusted the window size to accommodate the message log and buttons

        # Message log
        self.message_log = scrolledtext.ScrolledText(self.root, height=18, width=50, state='disabled', bg='white')
        self.message_log.pack(padx=10, pady=10)
        
        # Upload Button
        #self.upload_button = tk.Button(self.root, text="Upload Specification", command=self.upload_file)
        #self.upload_button.pack(pady=10)
        
        # Confirm Button
        #self.confirm_button = tk.Button(self.root, text="Confirm", command=self.confirm_upload, state='disabled')
        #self.confirm_button.pack(pady=5)

        # Initialize with a message
        self.log_message("System", "Say 'hello xarm' to start and say 'end of command' to finish.")
        
        self.uploaded_file_path = None  # Store the path of the uploaded file

    def log_message(self, sender, message):
        self.message_log.configure(state='normal')
        self.message_log.insert(tk.END, f"{sender}: {message}\n")
        self.message_log.configure(state='disabled')
        self.message_log.yview(tk.END)

    def start_ui(self):
        self.root.mainloop()

    '''
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Specification File",
            filetypes=(("PDF Files", "*.pdf"), ("All Files", "*.*"))
        )
        if file_path:
            self.uploaded_file_path = file_path
            self.log_message("System", f"Uploaded file: {os.path.basename(file_path)}")
            self.confirm_button.config(state='normal')

    def confirm_upload(self):
        if self.uploaded_file_path:
            self.log_message("System", "Upload complete.")
            self.process_file(self.uploaded_file_path)
            
    def process_file(self, file_path):
        
        pass
        # Process the specification content and send it to the robot agent
        # For now, we will just log it to the UI
        #self.log_message("System", content)
        #with open(self.uploaded_file_path, 'r') as file:
        #file_content = file.read()
    '''

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def log_message_to_ui(cls, sender, message):
        instance = cls.get_instance()
        instance.log_message(sender, message)

    @classmethod
    def start_ui_loop(cls):
        instance = cls.get_instance()
        instance.start_ui()
