#!/usr/bin/env python3
"""
Agent wrapping the OpenAI API
* Supports calling Python functions
* Supports nested architectures
"""
from __future__ import annotations

import logging
import robot_utils
from typing import Union
from llm_agent import LlmAgent
import threading
from voice_control import VoiceControl
from user_interface import UserInterface

# Create a logger
global_logger = logging.getLogger(__name__)
global_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('llm-roboticarm/log/global.log', mode='a')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
global_logger.addHandler(file_handler)
global_logger.addHandler(console_handler)

class User:
    """ Serves as entry point for human users.
    """
    def __init__(self) -> None:
        self.name = "user"
        self.annotation = "An agent that serves as an entry point for a user."
        self.inbox = []
        self.task_states = {}
        self.command = []
        self.voice_control = VoiceControl()  # Assuming VoiceControl is initialized without arguments

        #### Set up agent-specific logger ####
        self.logger = logging.getLogger(f'agent_{self.name}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        file_handler = logging.FileHandler(f'llm-roboticarm/log/{self.name}_actions.log', mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        ########################################
        
    def message(self, sender: str, message: list[tuple[str, str]]) -> str:
        self.inbox.append([(sender, message)])
        return "User received a message."

    def _speak_and_log(self, message: str, sender: str):
        threads = [
            threading.Thread(target=self.voice_control.text_to_speech, args=(message,)),
            threading.Thread(target=UserInterface.log_message_to_ui, args=(sender, message))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def run(self, peers: list[Union[LlmAgent, User]]) -> None:
        #TODO Make Inbox
        if self.inbox:
            for new in self.inbox:
                for sender, content in new:
                    task_identifier = robot_utils.get_task_identifier(sender, content)
                    if self.task_states.get(task_identifier) not in ['running', 'completed']:
                        self.logger.info(f"Running agent: {self.name}")
                        self.task_states[task_identifier] = 'running'
                        for sender, message in new:
                            threading.Thread(target=self._speak_and_log, args=(message, sender)).start()
                            self.inbox.pop(0)
                            self.task_states.pop(task_identifier, None)
                            
        if self.command:
            for new in self.command:
                command_identifier = robot_utils.get_command_identifier(new)       
                if self.task_states.get(command_identifier) not in ['running', 'completed']:
                    self.task_states[command_identifier] = 'running'
                    for peer in peers:
                            if len(peer.inbox) != 0:
                                if peer.inbox[-1][-1][0] != 'user':
                                    existing_content = peer.inbox[-1][-1][1]
                                    new_content = existing_content + "; " + new
                                    peer.inbox[-1][-1] = ("user", new_content)
                                else:
                                    peer.inbox.pop(0)
                                    peer.inbox.append([("user", new)])
                            else:
                                peer.inbox.append([("user", new)])
                            print(peer.inbox)
                            self.command.pop(0)
                            self.task_states.pop(command_identifier, None)
                            