
#!/usr/bin/env python3
"""
Agent wrapping the LangChain API
* Supports calling Python functions
* Supports nested architectures
"""
from __future__ import annotations

import json
import logging
import os
import threading
import utils

from typing import Union

import openai
import time

from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_ROBOT_AGENT, BASE_INSTRUCTIONS
from voice_control import VoiceControl
from user_interface import UserInterface

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

class LlmAgent:
    def __init__(
        self,
        model: str = None,
        name: str = None,
        annotation: str = None,
        instructions: str = None,
        functions_: list = None,
    ) -> None:
        ############################ Set up agent-specific logger ############################
        self.logger = logging.getLogger(f'agent_{name}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        file_handler = logging.FileHandler(f'llm-roboticarm/log/{name}_actions.log', mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        ######################################################################################

        self.model = "gpt-4o" if model is None else model
        self.name = name
        self.annotation = annotation
        self.function_analyzer = FunctionAnalyzer()
        self.function_info = []
        self.executables = {}
        self.inbox = []

        if functions_:
            self.function_info = [self.function_analyzer.analyze_function(f) for f in functions_]
            self.executables = {f.__name__: f for f in functions_}
        if self.function_info:
            if instructions:
                self.instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS + instructions
            else:
                self.instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS
            self.logger.info(f"Using function info:\n{json.dumps(self.function_info, indent=4)}")
        else:
            self.instructions = PROMPT_ROBOT_AGENT

        self.logger.info(
            f"Initialized with instructions:\n{self.instructions}"
        )

    def setup_message_functions(self, peers: list = None):
        for peer in peers:
            _description_extended = (
                f"This function messages a user, whose description is as follows:\n"
                f"{peer.annotation}\n"
                f"You can contact it by messaging it in natural language."
            )
            description = {
                "name": peer.name,
                "description": _description_extended,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sender": {
                            "type": "string",
                            "description": "The name of the sender (typically the user or agent's name).",
                        },
                        "message": {
                            "type": "string",
                            "description": "The message content that the peer agent should respond to.",
                        },
                    },
                    "required": ["sender", "message"],
                },
            }

            if not any(info['name'] == description['name'] for info in self.function_info):
                self.function_info.append(description)
                self.executables[peer.name] = getattr(peer, "message")

    def process_inbox(self):
        def handle_message(sender, content):
            inbox_identifier = utils.get_inbox_identifier(sender, content)
            if self.task_states.get(inbox_identifier) not in ['running', 'completed']:
                self.logger.info(f"Running agent: {self.name}")
                self.task_states[inbox_identifier] = 'running'

                msg_history_as_text = self._msg_history_to_prompt([(sender, content)])
                func_res, msgs = self.chat(msg_history_as_text)
                
                if 'func_type' in func_res:
                    status = self.chat_after_function_execution(func_res, msgs)
                    if status == "failed":
                        self.task_states[inbox_identifier] = 'failed'
                    else:
                        self.task_states[inbox_identifier] = 'completed'
                self.logger.info(f"Run for agent {self.name} done.")

        while True:
            if self.inbox:
                new_message = self.inbox.pop(0)
                for sender, content in new_message:
                    threading.Thread(target=handle_message, args=(sender, content)).start()
            time.sleep(0.1)

    def run(self, peers: list = None) -> None:
        self.setup_message_functions(peers)
        self.process_inbox()
        
class User:
    """ Serves as entry point for human users. """
    def __init__(self) -> None:
        self.name = "user"
        self.annotation = "An agent that serves as an entry point for a user."

        self.inbox = []
        self.tasks_states = {}

        self.commands = []
        self.command_states = {}

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

    def process_inbox(self, peers: list[Union[LlmAgent, User]]):
        while True:
            if self.inbox:
                new_message = self.inbox.pop(0)
                for sender, content in new_message:
                    inbox_identifier = utils.get_inbox_identifier(sender, content)
                    if self.tasks_states.get(inbox_identifier) not in ['running', 'completed']:
                        self.logger.info(f"Running agent: {self.name}")
                        self.tasks_states[inbox_identifier] = 'running'
                        self._speak_and_log(content, sender)
                        self.tasks_states.pop(inbox_identifier, None)
            time.sleep(0.1)

    def process_commands(self, peers: list[Union[LlmAgent, User]]):
        while True:
            if self.commands:
                new_command = self.commands.pop(0)
                command_identifier = utils.get_command_identifier(new_command)
                if self.command_states.get(command_identifier) not in ['running', 'completed']:
                    self.command_states[command_identifier] = 'running'
                    for peer in peers:
                        if peer.inbox:
                            if peer.inbox[-1][-1][0] != 'user':
                                existing_content = peer.inbox[-1][-1][1]
                                new_content = existing_content + "; " + new_command
                                peer.inbox[-1][-1] = ("user", new_content)
                            else:
                                peer.inbox.pop(0)
                                peer.inbox.append([("user", new_command)])
                        else:
                            peer.inbox.append([("user", new_command)])
                        self.logger.info(f"Sent command to {peer.name}: {new_command}")
                    self.command_states.pop(command_identifier, None)
            time.sleep(0.1)

    def run(self, peers: list[Union[LlmAgent, User]]) -> None:
        self.process_inbox(peers),
        self.process_commands(peers)

import utils
import functions

if __name__ == "__main__":
    # Define the file paths for the JSON files
    robot_file_path = 'llm-roboticarm/initialization/robots/'
    # Init Files
    robot_init_list = utils.get_init_files(robot_file_path)
    roboticarm_functions = functions.RoboticArmFunctions(robot_init_list)
    

    config = utils.load_json_data(robot_init_list[0])
    for name, robot_data in config.items():
        functions_ = []
        # Check if 'functions' key exists and is a list
        if 'functions' in robot_data and isinstance(robot_data['functions'], list):
            for function_name in robot_data['functions']:
            # Retrieve each function by name using getattr
                function_ref = getattr(roboticarm_functions, function_name, None)
                print(function_ref)
                print(functions_)
                if function_ref is not None:
                    functions_.append(function_ref)  # Append the function reference
                else:
                    print(f"Function '{function_name}' not found in functions module for agent {name}.")
        agent = LlmAgent(
            name=name, 
            annotation=robot_data.get('annotation', None),
            instructions=robot_data.get('instructions', None),
            functions_ = functions_
        )

    agent.run(list[User])
        
    
