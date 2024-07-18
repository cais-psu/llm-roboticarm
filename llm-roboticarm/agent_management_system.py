import logging
import agent_creator
from typing import Union
import threading
from llm_agent import LlmAgent, User

class AgentManagementSystem:
    def __init__(
        self,
        agents: list[Union[LlmAgent, User]],
        mas_dir: str,
    ) -> None:
        self.agents = agents
        self.queue = []
        self.mas_dir = mas_dir
        self.file_memory = []

    def run_agent(self, agent):
        peers = [a for a in self.agents if a is not agent]
        agent.run(peers=peers)

    def thread_start(self):
        for agent in self.agents:
            threading.Thread(target=self.run_agent, args=(agent,), daemon=True).start()
