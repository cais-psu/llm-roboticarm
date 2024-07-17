import os
import sys
import time
import openai
import re
import json
from rag_handler import RAGHandler
from langchain_openai import ChatOpenAI
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))
import utils
from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_ROBOT_AGENT, BASE_INSTRUCTIONS

class RoboticArmFunctions:
    def __init__(self, sop_file, params_file):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o")
        self.sop_handler = RAGHandler(sop_file, 'pdf', self.openai_api_key)
        self.sop_information = None
        self.params_information = utils.load_json_data(params_file)
        
        ########### list of robot functions for internal function call in perform_assembly_task function ###########
        #self.robot_functions = self.params_information.get("assembly_functions", [])
        #self.robot_function_analyzer = FunctionAnalyzer()
        #self.robot_function_info = [self.robot_function_analyzer.analyze_function(f) for f in self.robot_functions]
        #self.robot_executables = {f.__name__: f for f in self.robot_functions}
        #self.robot_instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS
        ############################################################################################################




    def _generated_params(self, task_description) -> str:
        """
        This function uses the LLM to generate parameters based on the task description and SOP information.

        :param task_description: The description of the task to generate parameters for
        :param sop_info: SOP information to base the parameter generation on
        """
        prompt = f"""
        Given the following task description, SOP information, and initial parameter information for assembly task, determine if the provided information is sufficient to perform the task. 
        If sufficient, modify the the corresponding parameter file to complete the task (without any description). 
        If insufficient, begin your response with "INSUFFICIENT" and explain why the information is inadequate, specifying what additional details are needed.

        Task Description:
        {task_description}

        SOP Information:
        {self.sop_information}

        Initial Parameter Information for Assembly:
        {self.params_information}

        Output Format:
        Sufficiency: [Yes/No]
        Generated Task or Needed Information:
        """

        msgs = [
            {"role": "system", "content": "You are a helpful assistant specialized in modifying parameter file for assembly operation based on provided task descriptions and SOP information."},
            {"role": "user", "content": prompt}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0.0,
        )

        print(response)

        return response.choices[0].message.content

    def perform_assembly_tasks(self, task_description: str) -> dict:
        """
        This function performs a task by retrieving SOP information, generating parameters, and executing the code.

        :param task_description: The description of the task to perform
        """
        # Step 1: Retrieve SOP information
        self.sop_information = self.provide_assembly_information(task_description)
        print(self.sop_information)

        # Step 2: Generate parameters
        generated_params = self._generated_params(f"Generate the parameters using the SOP and parameter information:\n{task_description}")
        print(generated_params)

        # Step 3: Execute the generated code
        # Parse the response to check sufficiency and either generate task or ask for more information
        if "INSUFFICIENT" in generated_params.upper():
            status = "failed"
            message = f"SOP information is not sufficient for the task: {task_description}. Additional information needed: {generated_task}"
        else:
            try:
                extracted_params = self._process_params(generated_params)
                print(extracted_params)

                #function call??

                status = "success"
                message = f"{task_description} executed successfully."
            except Exception as e:
                status = "failed"
                message = f"An error occurred while executing the task: {e}"

        result = {
            "func_type": "task",
            "status": status,
            "content": message,
        }

        return result
    
    def _process_params(self, generated_params: str) -> str:
        json_pattern = re.compile(r"json(.*?)```", re.DOTALL)
        match = json_pattern.search(generated_params)

        if match:
            json_content = match.group(1).strip()
            print("Extracted JSON content:")
            print(json_content)
        else:
            print("No JSON content found.")
        
        return json_content


    def provide_assembly_information(self, query: str) -> str:
        """
        This function provide assembly information from SOP using the RAG handler.

        :param query: The query to provide information for
        """
        message = self.sop_handler.retrieve(query)

        result = {
            "func_type": "info",
            "content": message,
        }
        return result




if __name__ == "__main__":
    sop_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\xArm_SOP.pdf"
    params_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\params.json"

    roboticarmfunction = RoboticArmFunctions(sop_file, params_file)
    roboticarmfunction.perform_assembly_tasks("human will do the housing step, you do the rest of the cable shark assembly process.")