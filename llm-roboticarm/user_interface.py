import tkinter as tk
from tkinter import scrolledtext

class UserInterface:
    _instance = None

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("xArm User Interface")
        self.root.geometry("1000x800")

        font_style = ("Arial", 24)

        # === Input Section at the Top ===
        separator = tk.Label(self.root, text="Enter Command:", font=("Arial", 20), anchor='w')
        separator.pack(fill=tk.X, padx=15, pady=(15, 0))

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        self.user_input = tk.Entry(self.input_frame, font=font_style)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.send_button = tk.Button(
            self.input_frame, text="Send", font=font_style, command=self.handle_text_input
        )
        self.send_button.pack(side=tk.RIGHT)

        self.user_input.bind("<Return>", lambda event: self.handle_text_input())

        # === Message Log Below, Fills Remaining Space ===
        self.message_log = scrolledtext.ScrolledText(
            self.root, state='disabled', bg='white', font=font_style
        )
        self.message_log.pack(padx=15, pady=(5, 15), fill=tk.BOTH, expand=True)

        self.log_message("System", "Say 'hello' or type a command to initiate.")

        # Must be set externally
        self.user = None
        self.roboticarm_agents = []
        self.user_input_control = None

    def handle_text_input(self):
        message = self.user_input.get().strip()
        if message:
            self.user_input.delete(0, tk.END)

            if not self.user_input_control or not self.user or not self.roboticarm_agents:
                self.log_message("System", "System not fully initialized.")
                return

            self.user_input_control.process_text_command(
                text_command=message,
                user=self.user,
                log_message=self.log_message,
                roboticarm_agents=self.roboticarm_agents
            )

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

# === TEST BLOCK ===
if __name__ == "__main__":
    class DummyUser:
        def __init__(self):
            self.commands = []

    class DummyRobotAgent:
        def message(self, sender, message):
            print(f"[DummyRobotAgent] Received from {sender}: {message}")

    class DummyInputControl:
        def process_text_command(self, text_command, user, log_message, roboticarm_agents):
            log_message("User", text_command)
            user.commands.append(text_command)
            for agent in roboticarm_agents:
                agent.message("user", text_command)

    ui = UserInterface.get_instance()
    ui.user = DummyUser()
    ui.roboticarm_agents = [DummyRobotAgent()]
    ui.user_input_control = DummyInputControl()
    ui.start_ui()
