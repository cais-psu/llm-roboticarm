import tkinter as tk
from tkinter import scrolledtext, filedialog
import os

class UserInterface:
    _instance = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("xArm User Interface")
        self.root.geometry("1000x800")  # Adjusted the window size to accommodate the message log and buttons

        # Set font style
        font_style = ("Arial", 30)  # Adjust font size

        # Message log with increased size and font
        self.message_log = scrolledtext.ScrolledText(
            self.root, height=22, width=70, state='disabled', bg='white', font=font_style
        )
        self.message_log.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        # Initialize with a message
        self.log_message("System", "Say 'hello' to initiate the command.")

        self.uploaded_file_path = None  # Store the path of the uploaded file

    def log_message(self, sender, message):
        self.message_log.configure(state='normal')
        self.message_log.insert(tk.END, f"{sender}: {message}\n")
        self.message_log.configure(state='disabled')
        self.message_log.yview(tk.END)

    def start_ui(self):
        self.root.mainloop()

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
