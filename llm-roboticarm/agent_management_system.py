import logging
import agent_creator
from typing import Union
import threading
from llm_agent import LlmAgent, User

class AgentManagementSystem:
    """
    A system for managing and coordinating multiple agents, either LLM-based agents 
    or user agents, within a multi-agent system (MAS) directory. This class provides 
    functionalities to initialize agents, handle task queues, and manage threads for 
    concurrent agent execution.

    Attributes:
        agents (list): A list of agents, either LlmAgent or User, managed by this system.
        mas_dir (str): The directory path for the multi-agent system.
        queue (list): A list to store tasks or messages queued for processing by agents.
        file_memory (list): A list to keep track of files or data associated with the agents.
    """

    def __init__(
        self,
        agents: list[Union[LlmAgent, User]],
        mas_dir: str,
    ) -> None:
        """
        Initializes the AgentManagementSystem with a list of agents and the MAS directory path.

        Args:
            agents (list): A list of agents (either LlmAgent or User) to be managed by the system.
            mas_dir (str): The directory path for the multi-agent system (MAS) data.
        """
        self.agents = agents
        self.queue = []  # Initialize an empty queue for tasks/messages
        self.mas_dir = mas_dir  # Set the directory path for the MAS
        self.file_memory = []  # Initialize an empty list for file memory tracking

    def run_agent(self, agent):
        """
        Executes an agent's task within the system, allowing the agent to interact
        with other agents as peers.

        Args:
            agent (Union[LlmAgent, User]): The agent to be executed within the MAS.
        """
        # Create a list of peers, excluding the agent itself
        peers = [a for a in self.agents if a is not agent]
        # Run the agent, passing in its peers for interaction
        agent.run(peers=peers)

    def thread_start(self):
        """
        Starts a separate thread for each agent in the system, allowing agents to run concurrently.
        Each agent is executed as a daemon thread.
        """
        # Loop over each agent and start a daemon thread to run it
        for agent in self.agents:
            threading.Thread(target=self.run_agent, args=(agent,), daemon=True).start()
