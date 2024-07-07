import tkinter as tk
from tkinter import scrolledtext

class UserInterface:
    _instance = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robotic Arm Voice Control")
        self.root.geometry("500x300")  # Adjusted the window size to accommodate the message log

        # Message log
        self.message_log = scrolledtext.ScrolledText(self.root, height=18, width=50, state='disabled', bg='white')
        self.message_log.pack(padx=10, pady=10)
        
        # Initialize with a message
        self.log_message("System", "Say 'hello xarm' to start and say 'end of command' to finish.")

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
