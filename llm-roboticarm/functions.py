import os
import sys
import time
import openai

from rag_handler import RAGHandler
from langchain_openai import ChatOpenAI
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))

class RoboticArmFunctions:
    def __init__(self, sop_file, code_file):
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o")
        self.sop_handler = RAGHandler(sop_file, 'pdf', self.openai_api_key)
        #self.code_handler = RAGHandler(code_file, 'py', self.openai_api_key)
        self.code_file = code_file
        
    def _generate_task(self, task_description: str, sop_information: str, code_information: str) -> str:
        """
        This function uses the LLM to generate a task based on the task description and SOP information.

        :param task_description: The description of the task to generate code for
        :param sop_info: SOP information to base the code generation on
        """
        prompt = f"""
        Given the following task description, SOP information, and Python code information, determine if the provided information is sufficient to perform the task. 
        If sufficient, generate only the corresponding Python code to complete the task (without any description). 
        If insufficient, begin your response with "INSUFFICIENT" and explain why the information is inadequate, specifying what additional details are needed.

        Task Description:
        {task_description}

        SOP Information:
        {sop_information}

        Python Code Information:
        {code_information}

        Output Format:
        Sufficiency: [Yes/No]
        Generated Task or Needed Information:
        """

        print(prompt)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant specialized in generating Python code based on provided task descriptions and SOP information."},
            {"role": "user", "content": prompt}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0.0,
        )

        print(response)

        return response.choices[0].message.content

    def perform_task(self, task_description: str) -> dict:
        """
        This function performs a task by retrieving SOP information, generating code, and executing the code.

        :param task_description: The description of the task to perform
        :param task_context: The context in which to execute the task
        """
        # Step 1: Retrieve SOP information
        sop_information = self.retrieve_information(task_description)
        print(sop_information)

        # Step 2: Retrieve Code information
        code_information = self._extract_text_from_py(self.code_file)
        print(code_information)

        # Step 3: Generate code
        generated_task = self._generate_task(f"Perform the following task using the SOP and code information:\n{task_description}", sop_information, code_information)
        print(generated_task)

        # Step 3: Execute the generated code
        # Parse the response to check sufficiency and either generate task or ask for more information
        if "INSUFFICIENT" in generated_task.upper():
            return {
                "status": "failed",
                "message": f"SOP information is not sufficient for the task: {task_description}. Additional information needed: {generated_task}"
            }
        else:
            try:
                exec(generated_task)
                task_result = {"status": "success", "message": "Task executed successfully."}
            except Exception as e:
                task_result = {"status": "failed", "message": f"An error occurred while executing the task: {e}"}

            return task_result

    def retrieve_information(self, query: str) -> str:
        """
        This function retrieves information using the RAG handler.

        :param query: The query to retrieve information for
        """
        return self.sop_handler.retrieve(query)

    def _extract_text_from_py(self, code_file_path):
        text = ''
        try:
            with open(code_file_path, 'r') as file:
                text = file.read()
        except Exception as e:
            print(f"An error occurred while reading the Python file: {e}")
        return text
    '''
    def _retrieve_code(self, query: str) -> str:
        """
        This function retrieves information using the RAG handler.

        :param query: The query to retrieve code for
        """
        final_query = "Generate python code based on the robotic_arm_assembly python file for the following process:" + query

        return self.code_handler.retrieve(final_query)
    '''


if __name__ == "__main__":
    sop_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\xArm_SOP.pdf"
    code_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\robotic_arm_assembly_old.py"

    roboticarmfunction = RoboticArmFunctions(sop_file, code_file)
    roboticarmfunction.perform_task("start the cable shark assembly process.")