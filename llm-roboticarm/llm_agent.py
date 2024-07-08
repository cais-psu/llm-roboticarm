#!/usr/bin/env python3
"""
Agent wrapping the OpenAI API
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
        non_function_model: str = "gpt-4o",
    ) -> None:
        """
        :param functions_: List of available functions
        :param class_instance_tuple: Tuple containing the class and the instance
        """
        # Set up agent-specific logger
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

        self.model = "gpt-4o" if model is None else model
        self.name = name
        self.annotation = annotation
        self.non_function_model = non_function_model
        self.function_analyzer = FunctionAnalyzer()
        self.function_info = []
        self.executables = {}

        self.inbox = []

        self.task_states = {}
        self.tasks = []
        self.queries = []
        self.query_states = []

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

    def message(self, sender: str, message: list[tuple[str, str]]) -> str:
        self.inbox.append([(sender, message)])
        return f"{self.name} Agent received a message."

    def _msg_history_to_prompt(self, msg_history: list[tuple[str, str]]) -> str:
        """ Turn a message history as it is stored in the query into a prompt history to be continued by the LLM
        Doing this as plain text allows having several agents in the conversation.
        """
        self.logger.info(f"{msg_history=}")
        messages = "\n".join([" The requester is " + m[0] + ". The requester " + m[0] + " sent this message: " + m[1] for m in msg_history])
        msg_history_as_text = (
            "This is a conversation history between several agents:\n"
            f"{messages}\n"
            f"Act as the agent {self.name} and return an appropriate response.\n"
        )
        return msg_history_as_text
    
    def __get_response(
        self,
        msgs: list[dict[str, str]],
        model: str,
        with_functions: bool = True,
        temperature: float = 0.0,
        function_call: dict = {}
    ) -> openai.openai_object.OpenAIObject:
        response = None

        #Sort the function so that message functions are recognized by LLM
        own_function = self.function_info.pop(0)
        self.function_info.append(own_function)

        while not response:
            try:
                if function_call:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msgs,
                        functions=self.function_info,
                        temperature=temperature,
                        function_call=function_call,
                    )
                elif with_functions:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msgs,
                        functions=self.function_info,
                        temperature=temperature,
                        function_call="auto",
                    )
                else:
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=msgs,
                        temperature=temperature,
                    )
            except openai.APIError as e:
                self.logger.error(e)
            except openai.RateLimitError as e:
                self.logger.error(e)

        return response

    def setup_message_functions(self, peers: list[LlmAgent]):
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

    def chat(self, prompt: str, model: str = None, temperature: float = 0.0) -> str:
        # choose the right model
        with_functions = len(self.function_info) > 0
        if model:
            model_ = model
        elif with_functions:
            model_ = self.model
        else:
            model_ = self.non_function_model

        msgs = []
        if self.instructions:
            msgs.append({"role": "system", "content": self.instructions})
        msgs.append({"role": "user", "content": prompt})

        self.logger.info(f"Msgs: {msgs}")
        response = self.__get_response(msgs=msgs, model=model_, with_functions=with_functions, temperature=temperature)
        self.logger.info(response)

        msgs.append(response.choices[0].message)

        try:
            if response.choices[0].finish_reason == "stop":
                func = "user"
                message = response.choices[0].message.content
                args = {"sender":"","message": message}
                self.logger.info(f"Function call: {func}; Arguments: {args}")
                
            else:
                func = response.choices[0].message.function_call.name
                args = json.loads(response.choices[0].message.function_call.arguments)
                self.logger.info(f"Function call: {func}; Arguments: {args}")

            # execute python 'message' or other actionable functions
            func_res = self.executables[func](**args)

            self.logger.info(f"Function returned `{func_res}`.")

        except KeyError:
            # This exception is raised when there's no function call in the response
            if response.choices[0].finish_reason == "stop":
                if 'completed' in response.choices[0].message.content:
                    #Clear the completed task
                    self.tasks.clear()
                else:
                    self.logger.info("No further function call to execute")

        except Exception as e:
            self.logger.error(f"An error occurred during function execution: {e}")            

        return func_res, msgs

    def chat_after_function_execution(self, func_res: dict, msgs: list, model: str = None, temperature: float = 0.0) -> str:
        # choose the right model
        with_functions = len(self.function_info) > 0
        if model:
            model_ = model
        elif with_functions:
            model_ = self.model
        else:
            model_ = self.non_function_model

        # Manufacturing process is executed and completed from the previous chat function
        if func_res['func_type'] == "assembly_process":
            if func_res['step_already_done'] == "completed":
                # get response from LLM
                func_msg = {
                    "role": "function",
                    "name": "xArm",
                    "content": func_res['content'],
                }
                msgs.append(func_msg)
                self.logger.info(f"Msgs: {msgs}")
                
                function_call = {"name": f"user"}
            else:
                func_msg = {
                    "role": "function",
                    "name": "xArm",
                    "content": f"{func_res['content']}, step_already_done: {func_res['step_already_done']}",
                }
                msgs.append(func_msg)
                self.logger.info(f"Msgs: {msgs}")
                
                function_call = {"name": f"user"}
        else:
            # get response from LLM
            func_msg = {
                "role": "function",
                "name": "xArm",
                "content": func_res['content'],
            }
            msgs.append(func_msg)
            self.logger.info(f"Msgs: {msgs}")
            
            function_call = {"name": f"user"}

        response = self.__get_response(msgs=msgs, model=model_, with_functions=True, temperature=temperature, function_call=function_call)
        self.logger.info(response)

        msgs.append(response.choices[0].message)
        try:
            func = response.choices[0].message.function_call.name
            args = json.loads(response.choices[0].message.function_call.arguments)
            
            # execute python 'message' function
            self.executables[func](**args)
            self.logger.info(f"Function call: {func}; Arguments: {args}")

            if {func_res['step_already_done']} == "completed":
                #Clear the completed task
                self.tasks.pop(0)
            else:
                self.tasks.pop(0)
                self.tasks.append([(func_res['robot_name'], f"{func_res['content']}, step_already_done: {func_res['step_already_done']}")])
                return "failed"
                
        except KeyError:
            if response.choices[0].finish_reason == "stop":
                if 'completed' in response.choices[0].message.content:
                    #Clear the completed task
                    self.tasks.clear()
                else:
                    self.logger.info("No further function call to execute")
        except Exception as e:
            self.logger.error(f"An error occurred during function execution: {e}")  

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

    def run(self, peers: list[Union[LlmAgent, User]]) -> None:
        self.setup_message_functions(peers)
        threading.Thread(target=self.process_inbox).start()

    '''
    def run(self, peers: list[Union[LlmAgent, User]]) -> None:
        if self.inbox:
            for new in self.inbox:
                for sender, content in new:
                    if sender == 'user':
                        inbox_identifier = utils.get_inbox_identifier(sender, content)
                        if self.task_states.get(inbox_identifier) not in ['running', 'completed']:
                            self.logger.info(f"Running agent: {self.name}")
                            self.setup_message_functions(peers)
                            self.task_states[inbox_identifier] = 'running'
                            msg_history_as_text = self._msg_history_to_prompt(new)
                            func_res, msgs = self.chat(msg_history_as_text)
                            if 'func_type' in func_res:
                                status = self.chat_after_function_execution(func_res, msgs)
                                if status == "failed":
                                    self.task_states[inbox_identifier] = 'failed'
                                else:
                                    self.task_states[inbox_identifier] = 'completed'        
                            self.logger.info(f"Run for agent {self.name} done.")
        else:
            pass
    '''

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

'''
class User:
    """ Serves as entry point for human users.
    """
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

    def run(self, peers: list[Union[LlmAgent, User]]) -> None:
        if self.inbox:
            for new in self.inbox:
                for sender, content in new:
                    inbox_identifier = utils.get_inbox_identifier(sender, content)
                    if self.command_states.get(inbox_identifier) not in ['running', 'completed']:
                        self.logger.info(f"Running agent: {self.name}")
                        self.tasks_states[inbox_identifier] = 'running'
                        for sender, message in new:
                            threading.Thread(target=self._speak_and_log, args=(message, sender)).start()
                            self.inbox.pop(0)
                            self.tasks_states.pop(inbox_identifier, None)
                            
        if self.commands:
            for new in self.commands:
                command_identifier = utils.get_command_identifier(new)       
                if self.command_states.get(command_identifier) not in ['running', 'completed']:
                    self.command_states[command_identifier] = 'running'
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
                            self.commands.pop(0)
                            self.command_states.pop(command_identifier, None)        
'''                            