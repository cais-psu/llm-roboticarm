import os
import sys
import openai
import re
import json
from rag_handler import RAGHandler
from langchain_openai import ChatOpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))

import general_utils
from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_ROBOT_AGENT, BASE_INSTRUCTIONS, LOG_RETRIEVAL_INSTRUCTIONS
import robotic_arm_assembly
from datetime import datetime

class RoboticArmFunctions:
    def __init__(self, sop_file, params_general_path, params_movement_path):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o")
        self.sop_handler = RAGHandler(sop_file, 'pdf', self.openai_api_key)
        self.sop_information = None
        self.params_general_json = general_utils.load_json_data(params_general_path)
        self.params_movement = params_movement_path
        self.log_file_path = 'llm-roboticarm/log/xArm_actions.log'
        
    def _generated_params(self, task_description) -> str:
        """
        This function uses the LLM to generate parameters based on the task description and SOP information.

        :param task_description: The description of the task to generate parameters for
        """
        prompt = f"""
        Given the following task description, SOP information, and initial parameter and configuration information for assembly task,
        modify the the corresponding parameter file to complete the task (without any description). 

        Instructions:
        1. If the task description specifies that certain assembly steps will be completed by humans or should be skipped, remove those steps from the "assembly_steps" list.
        2. Ensure the remaining steps in the "assembly_steps" list reflect the order in which the cable shark assembly should proceed based on the task description.
        3. The "assembly_functions" list should remain unchanged unless explicitly stated in the task description.
        4. Do not include any additional explanations or descriptions in the output, only provide the modified parameter file in JSON format.

        Task Description:
        {task_description}

        SOP Information:
        {self.sop_information}

        Specific Parameters or Configurations for Assembly in JSON:
        {self.params_general_json}

        Output Format:
        Generated Parameter File in JSON:
        """

        msgs = [
            {"role": "system", "content": "You are a helpful assistant specialized in modifying parameter file for assembly operation based on provided task descriptions, SOP information, and JSON parameter information."},
            {"role": "user", "content": prompt}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0.0,
        )

        return response.choices[0].message.content

    def perform_tasks(self, task_description: str, step_working_on: str) -> dict:
        """
        This function performs a task by retrieving SOP information, generating parameters, and executing the code.

        :param task_description: The description of the task to perform
        :param step_working_on: name of the assembly step that is working on
        """
        # Step 1: Retrieve SOP information
        self.sop_information = self.provide_information_or_message(task_description)
        print(self.sop_information)

        # Step 2: Generate parameters
        generated_params_general = self._generated_params(f"Generate the parameters using the SOP and parameter information:\n{task_description}")
        print(generated_params_general)

        # Step 3: Execute the generated code
        # Parse the response to check sufficiency and either generate task or ask for more information
        try:
            self.new_params_general = self._process_params(generated_params_general)

            ########### list of robot functions for internal function call in perform_assembly_task function ###########
            self.assembly = robotic_arm_assembly.RoboticArmAssembly(self.new_params_general, self.params_movement)
            self.assembly_functions = self.params_general_json.get("assembly_functions", [])

            self.assembly_functions_ = []
            for assembly_function_name in self.assembly_functions:
                assembly_function_ref = getattr(self.assembly, assembly_function_name)
                self.assembly_functions_.append(assembly_function_ref)  
                
            # Append the function reference
            self.assembly_function_analyzer = FunctionAnalyzer()
            self.assembly_function_info = [self.assembly_function_analyzer.analyze_function(f) for f in self.assembly_functions_]
            self.assembly_executables = {f.__name__: f for f in self.assembly_functions_}
            self.assembly_instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS

            self.assembly_prompt = f"""
            Task Description:
            {task_description}

            SOP Information:
            {self.sop_information}

            New Parameter Information for Assembly:
            {self.new_params_general}

            step_working_on:
            {step_working_on}
            """

            msgs = []
            msgs.append({"role": "system", "content": self.assembly_instructions})
            msgs.append({"role": "user", "content": self.assembly_prompt})

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=msgs,
                functions=self.assembly_function_info,
                temperature=0.0,
                function_call="auto",
            )

            print(response)
            func = response.choices[0].message.function_call.name
            args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"Function call: {func}; Arguments: {args}")

            # execute assembly functions
            step_working_on, message = self.assembly_executables[func](**args)

        except Exception as e:
            step_working_on = None
            message = f"An error occurred while executing the task: {e}"

        result = {
            "func_type": "task",
            "step_working_on": step_working_on,
            "content": message
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

    def provide_status(self, status_query: str) -> str:
        """
        This function provides the current status and what the robot has been doing based on retrieving log history data.

        :param status_query: The status query to provide status for
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_handler = RAGHandler(self.log_file_path, 'txt', self.openai_api_key)
        message = self.log_handler.retrieve(f"Status Query: {status_query}, Current Time: {current_time}, LOG_RETRIEVAL_INSTRUCTIONS: {LOG_RETRIEVAL_INSTRUCTIONS}")

        result = {
            "func_type": "info",
            "content": message,
        }
        return result
    
    def provide_information_or_message(self, query: str) -> str:
        """
        This function provide information or message based on SOP using the RAG handler.

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
    params_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\params_general.json"

    roboticarmfunction = RoboticArmFunctions(sop_file, params_file)
    #roboticarmfunction.perform_assembly_tasks("human will do the housing step, you do the rest of the cable shark assembly process.")
    roboticarmfunction.perform_assembly_tasks("Execute the cable shark assembly process.")