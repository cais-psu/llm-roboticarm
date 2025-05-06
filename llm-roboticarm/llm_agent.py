
#!/usr/bin/env python3
"""
Agent wrapping the LangChain API
* Supports calling Python functions
* Supports nested architectures
"""
from __future__ import annotations

import json
#import logging
import log_setup

import os
import threading
import general_utils

from typing import Union

import openai
import time

from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_ROBOT_AGENT, BASE_INSTRUCTIONS
from user_input_control import UserInputControl
from user_interface import UserInterface

# load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

class LlmAgent:
    """
    Represents an AI-powered agent that can interact with users and other agents, 
    process instructions, execute functions, and log actions.
    """    
    def __init__(
        self,
        model: str = None,
        name: str = None,
        annotation: str = None,
        instructions: str = None,
        functions_: list = None,

    ) -> None:
        """
        Initializes the LlmAgent class.

        Parameters
        ----------
        model : str, optional
            The model to be used for generating responses (default is "gpt-4o").
        name : str, optional
            The name of the agent.
        annotation : str, optional
            Additional information about the agent.
        instructions : str, optional
            Initial instructions for the agent.
        functions_ : list, optional
            List of Python functions that the agent can execute.
        """        

        # Set up logging for both agent-specific and action logs
        self.log_setup = log_setup.LogSetup(name="xArm")
        self.log_setup.setup_logging("agent")
        self.log_setup.setup_logging("action")
        self.logger_agent = self.log_setup.logger_agent
        self.logger_action = self.log_setup.logger_action

        self.model = "gpt-4o" if model is None else model
        self.name = name
        self.annotation = annotation

        self.function_analyzer = FunctionAnalyzer()
        self.function_info = []
        self.executables = {}

        self.message_function_info = []
        self.message_executables = {}

        self.inbox = []
        self.task_states = {}
        self.tasks = []        
        
        self.msg_history_as_text = ""
        
        if functions_:
            self.function_info = [self.function_analyzer.analyze_function(f) for f in functions_]
            self.executables = {f.__name__: f for f in functions_}
        if self.function_info:
            if instructions:
                self.instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS + instructions
            else:
                self.instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS
        else:
            self.instructions = PROMPT_ROBOT_AGENT

        self.logger_agent.info(
            f"Initialized with instructions:\n{self.instructions}"
        )

        # Log that the xArm agent has been initialized and the status is idle
        self.logger_action.info("The xArm has been initialized and the status is idle.")
        
    def message(self, sender: str, message: list[tuple[str, str]]) -> str:
        """
        Adds a message to the inbox for processing.
        
        Parameters
        ----------
        sender : str
            The sender of the message.
        message : list[tuple[str, str]]
            Message content in list of tuples format.
        
        Returns
        -------
        str
            Confirmation message indicating receipt.
        """        
        self.inbox.append([(sender, message)])
        return f"{self.name} Agent received a message."
    
    def setup_message_functions(self, peers: list = None):
        """
        Sets up messaging functions for communication with peer agents.

        Parameters
        ----------
        peers : list
            List of peer agents for inter-agent communication.
        """        
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

            if not any(info['name'] == description['name'] for info in self.message_function_info):
                self.message_function_info.append(description)
                self.message_executables[peer.name] = getattr(peer, "message")

    def _msg_history_to_prompt(self, msg_history: list[tuple[str, str]]) -> str:
        """
        Converts message history into a prompt for the language model.

        Parameters
        ----------
        msg_history : list[tuple[str, str]]
            List of messages in (sender, content) format.

        Returns
        -------
        str
            Prompt formatted as message history.
        """
        self.logger_agent.info(f"{msg_history=}")
        messages = "\n".join([" The requester is " + m[0] + ". The requester " + m[0] + " sent this message: " + m[1] for m in msg_history])
        msg_history_as_text = (
            #"This is a conversation history between human and robot:\n"
            f"{messages}\n"
            #f"Act as the agent {self.name} and return an appropriate response.\n"
        )
        return msg_history_as_text
    
    def __get_function_call_response(
        self,
        msgs: list[dict[str, str]],
        model: str,
        with_functions: bool = True,
        temperature: float = 0.0,
        function_call: dict = {}
    ) -> openai.openai_object.OpenAIObject:
        """
        Requests a response from OpenAI API with optional function call.

        Parameters
        ----------
        msgs : list
            Messages to include in the API call.
        model : str
            Model to use for the call.
        with_functions : bool, optional
            Whether to enable functions (default is True).
        temperature : float, optional
            Controls randomness (default is 0.0).
        function_call : dict, optional
            Specific function call configuration (default is empty dict).
        
        Returns
        -------
        openai.openai_object.OpenAIObject
            The API response.
        """        
        response = None

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
            except openai.APIError as e:
                self.logger_agent.error(e)
            except openai.RateLimitError as e:
                self.logger_agent.error(e)

        return response    

    def __get_message_response(
        self,
        msgs: list[dict[str, str]],
        model: str,
        with_functions: bool = True,
        temperature: float = 0.0
    ) -> openai.openai_object.OpenAIObject:
        response = None
        while not response:
            try:
                if with_functions and self.message_function_info:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msgs,
                        functions=self.message_function_info,
                        temperature=temperature
                    )
                else:
                    response = openai.chat.completions.create(
                        model=model,
                        messages=msgs,
                        temperature=temperature
                    )
            except openai.error.InvalidRequestError as e:
                self.logger_agent.error(f"Error code: {e.http_status} - {e.json_body}")
                break  # Exit the loop on unrecoverable error
            except openai.APIError as e:
                self.logger_agent.error(e)
            except openai.RateLimitError as e:
                self.logger_agent.error(e)
        return response
    
    
    def chat(self, prompt: str, model: str = None, temperature: float = 0.0) -> str:
        """
        Sends a chat message to the language model and processes the response.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        model : str, optional
            Model to use for the call (default is self.model).
        temperature : float, optional
            Controls randomness of the response (default is 0.0).
        
        Returns
        -------
        tuple
            Result of the function and messages sent to the model.
        """
        
        with_functions = len(self.function_info) > 0
        if model:
            model_ = model
        elif with_functions:
            model_ = self.model

        msgs = []
        if self.instructions:
            msgs.append({"role": "system", "content": self.instructions})
        msgs.append({"role": "user", "content": prompt})

        self.logger_agent.info(f"Msgs: {msgs}")
        response = self.__get_function_call_response(msgs=msgs, model=model_, with_functions=with_functions, temperature=temperature)
        self.logger_agent.info(response)

        msgs.append(response.choices[0].message)
        func_res = {}
        try:
            if response.choices[0].finish_reason == "stop":
                func = "provide_information_or_message"
                message = response.choices[0].message.content
                args = {"sender":"","message": message}
                self.logger_agent.info(f"Function call: {func}; Arguments: {args}")
                func_res = {
                    "func_type": "info",
                    "content": message,
                    "func_name": func  # Include the function name if needed
                }
            else:
                func = response.choices[0].message.function_call.name
                args = json.loads(response.choices[0].message.function_call.arguments)
                self.logger_agent.info(f"Function call: {func}; Arguments: {args}")
                # execute python actionable functions
                func_res = self.executables[func](**args)
                func_res['func_name'] = func  # Include the function name in the result

            self.logger_agent.info(f"Function returned `{func_res}`.")

        except KeyError:
            # This exception is raised when there's no function call in the response
            if response.choices[0].finish_reason == "stop":
                if 'completed' in response.choices[0].message.content:
                    #Clear the completed task
                    self.tasks.clear()
                else:
                    self.logger_agent.info("No further function call to execute")

        except Exception as e:
            self.logger_agent.error(f"An error occurred during function execution: {e}")            

        return func_res, msgs
    
    def __get_direct_response(self, msgs, temperature=0.0):
        response = None
        while not response:
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=temperature
                )
            except openai.error.OpenAIError as e:
                print(f"Error: {e}")
        return response

    def chat_after_function_execution(self, func_res: dict, msgs: list, model: str = None, temperature: float = 0.0) -> None:
        func_name = func_res.get('func_name')

        # Check if the function is 'provide_status' or 'provide_information_or_message'
        if func_name in ['provide_status', 'provide_information_or_message']:

            # Prepare the function message
            func_msg = {
                "role": "function",
                "name": self.name,
                "content": func_res['content'],
            }
            msgs.append(func_msg)
            self.logger_agent.info(f"Msgs: {msgs}")

            # Get response from LLM again
            response = self.__get_direct_response(
                msgs=msgs, temperature=temperature
            )
            self.logger_agent.info(response)
            msgs.append(response.choices[0].message)

            args = {'sender': self.name, 'message': response.choices[0].message.content}

            # Send the message to the user
            user_messaging_function = self.message_executables.get('user')
            if user_messaging_function:
                user_messaging_function(**args)
            else:
                self.logger_agent.error("User messaging function not found.")
        else:
            # Proceed with the existing logic for other functions
            with_functions = len(self.function_info) > 0
            model_ = model or (self.model if with_functions else None)

            # Prepare the function message
            func_msg = {
                "role": "function",
                "name": self.name,
                "content": func_res['content'],
            }
            msgs.append(func_msg)
            self.logger_agent.info(f"Msgs: {msgs}")

            # Update message history
            self.msg_history_as_text += self._msg_history_to_prompt([(func_msg['name'], func_msg['content'])])

            # Get response from LLM again
            response = self.__get_message_response(
                msgs=msgs, model=model_, with_functions=True, temperature=temperature
            )
            self.logger_agent.info(response)
            msgs.append(response.choices[0].message)

            try:
                if response.choices[0].finish_reason == "stop":
                    func = response.choices[0].message.function_call.name
                    args = json.loads(response.choices[0].message.function_call.arguments)
                    self.logger_agent.info(f"Function call: {func}; Arguments: {args}")
                else:
                    func = response.choices[0].message.function_call.name
                    args = json.loads(response.choices[0].message.function_call.arguments)
                    self.logger_agent.info(f"Function call: {func}; Arguments: {args}")

                # Execute python 'message' functions
                func_res = self.message_executables[func](**args)
                self.logger_agent.info(f"Function returned `{func_res}`.")

            except KeyError:
                if response.choices[0].finish_reason == "stop":
                    if 'completed' in response.choices[0].message.content:
                        self.tasks.clear()
                    else:
                        self.logger_agent.info("No further function call to execute")
            except Exception as e:
                self.logger_agent.error(f"An error occurred during function execution: {e}")

    def process_inbox(self):
        """
        Processes incoming messages in the inbox,  oking the chat function.
        """        
        def handle_message(sender, content):
            inbox_identifier = general_utils.get_inbox_identifier(sender, content)
            if self.task_states.get(inbox_identifier) not in ['running', 'completed']:
                self.logger_agent.info(f"Running agent: {self.name}")
                self.task_states[inbox_identifier] = 'running'

                self.msg_history_as_text += self._msg_history_to_prompt([(sender, content)])
                func_res, msgs = self.chat(self.msg_history_as_text)

                # If 'func_type' exists, this means that it is not just simple message but function call
                if 'func_type' in func_res:
                    status = self.chat_after_function_execution(func_res, msgs)
                    self.task_states[inbox_identifier] = 'completed'

                self.logger_agent.info(f"Run for agent {self.name} done.")

        while True:
            if self.inbox:
                new_message = self.inbox.pop(0)
                for sender, content in new_message:
                    threading.Thread(target=handle_message, args=(sender, content)).start()
            time.sleep(0.1)

    def run(self, peers: list = None) -> None:
        """
        Starts the agent's inbox processing in a background thread.

        Parameters
        ----------
        peers : list
            List of peer agents to set up messaging functions.
        """        
        self.setup_message_functions(peers)
        threading.Thread(target=self.process_inbox).start()
        
class User:
    """ Serves as entry point for human users. """
    def __init__(self) -> None:
        self.name = "user"
        self.annotation = "An agent that serves as an entry point for a user."

        self.inbox = []
        self.tasks_states = {}

        self.commands = []
        self.command_states = {}

        self.user_input_control = UserInputControl()  # Assuming User Input Control is initialized without arguments

        # Set up agent-specific logger
        self.log_setup = log_setup.LogSetup(name="User")
        self.log_setup.setup_logging("agent")
        self.logger = self.log_setup.logger_agent

    def message(self, sender: str, message: list[tuple[str, str]]) -> str:
        """
        Adds a received message to the inbox for processing.

        Parameters
        ----------
        sender : str
            The name of the message sender.
        message : list[tuple[str, str]]
            The message content in a list of tuples.

        Returns
        -------
        str
            Confirmation message indicating the message was received.
        """        
        self.inbox.append([(sender, message)])
        return "User received a message."

    def _speak_and_log(self, message: str, sender: str):
        """
        Outputs a message through user input control and logs it to the UI.

        Parameters
        ----------
        message : str
            The message content.
        sender : str
            The sender's name for logging purposes.
        """        
        threads = [
            threading.Thread(target=self.user_input_control.text_to_speech, args=(message,)),
            threading.Thread(target=UserInterface.log_message_to_ui, args=(sender, message))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def process_inbox(self, peers: list[Union[LlmAgent, User]]):
        """Continuously processes incoming messages in the inbox."""
        while True:
            if self.inbox:
                new_message = self.inbox.pop(0)
                for sender, content in new_message:
                    inbox_identifier = general_utils.get_inbox_identifier(sender, content)
                    if self.tasks_states.get(inbox_identifier) not in ['running', 'completed']:
                        self.logger.info(f"Running agent: {self.name}")
                        self.tasks_states[inbox_identifier] = 'running'
                        self._speak_and_log(content, sender)
                        self.tasks_states.pop(inbox_identifier, None)
            time.sleep(0.1)

    def process_commands(self, peers: list[Union[LlmAgent, User]]):
        """Continuously processes commands, sending them to specified peers."""
        while True:
            if self.commands:
                new_command = self.commands.pop(0)
                command_identifier = general_utils.get_command_identifier(new_command)
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
        """
        Starts the agent's message and command processing in separate threads.

        Parameters
        ----------
        peers : list
            List of peer agents that can interact with this agent.
        """        
        self.process_inbox(peers),
        self.process_commands(peers)

import general_utils
import functions

if __name__ == "__main__":
    # Define the file paths for the JSON files
    robot_file_path = 'llm-roboticarm/initialization/robots/'
    # Init Files
    robot_init_list = general_utils.get_init_files(robot_file_path)
    roboticarm_functions = functions.RoboticArmFunctions(robot_init_list)
    
    config = general_utils.load_json_data(robot_init_list[0])
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
        
    
