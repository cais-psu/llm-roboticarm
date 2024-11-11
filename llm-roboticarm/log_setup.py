import logging
import os

class LogSetup:
    def __init__(self, name: str, log_dir: str = "llm-roboticarm/log"):
        """
        Initializes the LogSetup class, setting up loggers for agent, action, or process logs.

        Parameters
        ----------
        name : str
            The name used for naming log files and loggers.
        log_dir : str, optional
            Directory for storing log files (default is "llm-roboticarm/log").
        """
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

        # Initialize logger attributes as None
        self.logger_agent = None
        self.logger_action = None
        self.logger_process = None

    def setup_logging(self, logging_type: str):
        """
        Sets up a logger based on the specified logging type with file and console handlers.

        Parameters
        ----------
        logging_type : str
            The type of logger to set up. Must be 'agent', 'action', or 'process'.
        """
        # Define valid logging types and their formats
        log_types = {
            "agent": {"filename": f"{self.name}_agent.log", "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s'},
            "action": {"filename": f"{self.name}_actions.log", "format": '%(asctime)s - %(name)s - %(levelname)s - [ACTION] - %(message)s'},
            "process": {"filename": f"{self.name}_process.log", "format": '%(asctime)s - %(name)s - %(levelname)s - [PROCESS] - %(message)s'}
        }

        # Check for valid logging type
        if logging_type not in log_types:
            raise ValueError(f"Invalid logging type '{logging_type}'. Choose from 'agent', 'action', or 'process'.")

        # Set up the logger
        logger = logging.getLogger(f'{logging_type}_{self.name}')
        logger.setLevel(logging.INFO)

        # File handler setup
        file_handler = logging.FileHandler(os.path.join(self.log_dir, log_types[logging_type]["filename"]), mode='a')
        file_handler.setFormatter(logging.Formatter(log_types[logging_type]["format"]))
        logger.addHandler(file_handler)

        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_types[logging_type]["format"]))
        logger.addHandler(console_handler)

        # Assign the logger to the appropriate attribute based on logging type
        setattr(self, f"logger_{logging_type}", logger)

# Example usage
if __name__ == "__main__":
    log_setup = LogSetup(name="robot")

    # Setup each type of logging
    log_setup.setup_logging("agent")
    log_setup.setup_logging("action")
    log_setup.setup_logging("process")

    # Access and use the loggers
    log_setup.logger_agent.info("Agent logger message.")
    log_setup.logger_action.info("Action logger message.")
    log_setup.logger_process.info("Process logger message.")
