import os
import sys
import time

from rag_handler import RAGHandler
from langchain_openai import ChatOpenAI
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))

class RoboticArmFunctions:
    def __init__(self, specification_file):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_handler = RAGHandler(specification_file, self.openai_api_key)
        
    def generate_code(self, task_description: str) -> str:
        """
        This function uses the LLM to generate code based on the task description.

        :param task_description: The description of the task to generate code for
        """
        prompt = f"""
        Based on the following task description, generate the Python code to perform the task:

        Task Description:
        {task_description}

        Generated Code:
        """
        response = self.llm.generate(prompt)
        return response["choices"][0]["message"]["content"]

    def perform_task(self, task_description: str, task_context: dict) -> dict:
        """
        This function performs a task by retrieving SOP information, generating code, and executing the code.

        :param task_description: The description of the task to perform
        :param task_context: The context in which to execute the task
        """
        # Step 1: Retrieve SOP information
        sop_information = self.retrieve_information(task_description)
        
        # Step 2: Generate code
        generated_code = self.generate_code(f"Perform the following task using the SOP information:\n{sop_information}")

        # Step 3: Execute the generated code
        exec_globals = {}
        exec_locals = task_context
        try:
            exec(generated_code, exec_globals, exec_locals)
            task_result = exec_locals.get("result", {})
            if not task_result:
                task_result = {"status": "success", "message": "Task executed successfully."}
        except Exception as e:
            task_result = {"status": "failed", "message": f"An error occurred while executing the task: {e}"}
        
        return task_result

    def retrieve_information(self, query: str) -> str:
        """
        This function retrieves information using the RAG handler.

        :param query: The query to retrieve information for
        """
        return self.rag_handler.retrieve(query)
