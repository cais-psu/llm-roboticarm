import os
import sys
import openai
import re
import json
from rag_handler import RAGHandler
from langchain_openai import ChatOpenAI
from prompts import PROVIDE_INFORMATION_INSTRUCTIONS
import cv2
import base64
from openai import OpenAI
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK-master'))

import general_utils
from function_analyzer import FunctionAnalyzer
from prompts import PROMPT_ROBOT_AGENT, BASE_INSTRUCTIONS, LOG_RETRIEVAL_INSTRUCTIONS
import robot_tasks
from datetime import datetime

class RoboticArmFunctions:
    def __init__(self, specification_files, robot_config, product_config, robot_assembly, camera_manager):
        """
        specification_files: List of tuples [(filepath1, filetype1), (filepath2, filetype2), ...]
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.openai_api_key)
        self.specification_handler = RAGHandler(specification_files, self.openai_api_key)  # Modified to handle multiple files
        self.robot_config = robot_config
        self.product_config = product_config
        self.robot_assembly = robot_assembly
        self.camera_manager = camera_manager

    def _generated_params(self, task_description) -> str:
        """
        This function uses the LLM to generate parameters based on the task description and specification information.

        :param task_description: The description of the task to generate parameters for
        """
        prompt = f"""
        Given the following task description, specification information, and initial parameter and configuration information for the cable shark assembly task,
        modify ONLY the "assembly_steps" list in the parameter file to exactly match the steps requested in the task description.

        Instructions:
        1. If the task description specifies that only specific steps (e.g., "spring assembly") are to be performed, include only those in the "assembly_steps" list.
        2. Do NOT infer or add any additional steps such as 'cap' unless they are explicitly mentioned in the task description.
        3. If the description includes skipping or assigning steps to humans, remove those steps from the "assembly_steps" list.
        4. The "assembly_functions" list should remain unchanged unless explicitly instructed.
        5. Do not include any extra explanations or metadata—only the modified parameter file in pure JSON format.

        Task Description:
        {task_description}

        Specification Information:
        {self.specification_information}

        Assembly Sequences and Configurations in JSON:
        {self.product_config}

        Output Format:
        Modified Parameter File in JSON:
        """

        msgs = [
            {"role": "system", "content": "You are a helpful assistant specialized in modifying parameter file for assembly operation based on provided task descriptions, specification information, and JSON parameter information."},
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
        This function performs an assembly task by first determining whether the user's message is a request to perform an actionable task.
        If the message is confirmed as a task request, the function proceeds to:
        1. Generate new assembly parameters based on the user's request, specification, and product configuration.
        2. Execute the corresponding robotic function using the LLM's structured output to determine which function to call.
        3. If the message does not contain an actionable request (e.g., it's a question or comments), the function will not proceed and will return a response indicating no action was taken.

        :param task_description: The task-related user instruction provided at the end of the sentence immediately following "The requester user sent this message:". If multiple user messages exist, only the final command is considered, without summarization or modification.                        
        :param step_working_on: The specific step of the assembly to begin or resume from. If the request is for a general assembly task, this parameter should default to 'housing'. In the case of error recovery, this should reflect the last successfully completed or fixed step.
        """
        # Step 1: Retrieve Specification information
        self.specification_information = self.provide_information_or_message(task_description)
        print(self.specification_information)

        # Step 2: Generate parameters
        generated_product_config = self._generated_params(f"Generate the parameters using the specification and parameter information:\n{task_description}")
        print(generated_product_config)

        # Step 3: Execute the generated code
        # Parse the response to check sufficiency and either generate task or ask for more information
        try:
            self.new_product_config = self._process_params(generated_product_config)

            ########### list of robot functions for internal function call in perform_assembly_task function ###########
            self.assembly_functions = self.robot_config.get("assembly_functions", [])

            self.assembly_functions_ = []
            for assembly_function_name in self.assembly_functions:
                assembly_function_ref = getattr(self.robot_assembly, assembly_function_name)
                self.assembly_functions_.append(assembly_function_ref)  
                
            # Append the function reference
            self.assembly_function_analyzer = FunctionAnalyzer()
            self.assembly_function_info = [self.assembly_function_analyzer.analyze_function(f) for f in self.assembly_functions_]
            self.assembly_executables = {f.__name__: f for f in self.assembly_functions_}
            self.assembly_instructions = PROMPT_ROBOT_AGENT + BASE_INSTRUCTIONS

            self.assembly_prompt = f"""
            Task Description:
            {task_description}

            specification Information:
            {self.specification_information}

            New Parameter Information for Assembly:
            {self.new_product_config}

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
            
            func = response.choices[0].message.function_call.name
            args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"Function call: {func}; Arguments: {args}")

            # execute assembly functions
            self.robot_assembly.refresh_config(self.new_product_config)
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

    def provide_status(self, status_query: str, msg_history: str) -> dict:
        """
        Provides the robot's physical status using the raw string version of the message history.

        :param status_query: The user's status query (e.g., "what did you just do?")
        :param msg_history: The string-formatted previous message history (e.g., self.msg_history_as_text)
        """

        #print(msg_history)
        prompt = f"""
            You are a robot assistant. The user is asking for a status update.

            Message History:
            {msg_history}

            User Query:
            {status_query}

            Respond with what physical actions the robot took (e.g., assembled spring), based on the message history. 
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are summarizing the robot's physical actions based on user query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        return {
            "func_type": "info",
            "content": response.choices[0].message.content.strip(),
        }
    
    def provide_information_or_message(self, query: str) -> str:
        """
        This function provide information or message based on specification using the RAG handler.

        :param query: The full content provided by the user only at the end of the sentence right after "The requester user sent this message:". The sentence is usually in a question asking for information. If there are multiple requests from the user, ensure that the query contains only the **last command** without any summarization or alteration.
        """
        message = self.specification_handler.retrieve(query + PROVIDE_INFORMATION_INSTRUCTIONS)

        result = {
            "func_type": "info",
            "content": message,
        }
        return result

    def perceive_environment(self, image_query: str, msg_history: str) -> dict:
        """
        Captures a frame from the camera, saves it with a timestamp, sends it to GPT-4o with vision,
        and returns a concise description of the environment based on the image and message history.

        :param image_query: Final user instruction from the message, no summarization or rewriting.
        :param msg_history: The string-formatted previous message history (e.g., self.msg_history_as_text)
        """
        import datetime  # Ensure you import this at the top of your file if not already
        cam_type = 'cameraA'
        try:
            # Step 1: Capture and save image (timestamped)
            frame = self.camera_manager.capture_scene_with_detections(cam_type=cam_type)
            if frame is None:
                return {"func_type": "vlm", "content": f"Failed to capture image from {cam_type}."}

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join("captured_images", f"{cam_type}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            # Step 2: Encode image as base64
            with open(image_path, "rb") as img_file:
                base64_img = base64.b64encode(img_file.read()).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_img}"

            # Step 3: Compose the prompt
            prompt = f"""
            You are a robot assistant using vision and task memory to understand your current environment.
            Use the image and the history of tasks below to answer the user's request clearly.
            Be concise and specific in natural language. Do not restate the prompt.

            Message History:
            {msg_history}

            User Query:
            {image_query}

            Describe the scene: objects, their positions, or assembly state.
            """

            # Step 4: Send to GPT-4o
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You analyze robot camera images using visual and contextual information to help with perception and assembly."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=300
            )

            return {
                "func_type": "vlm",
                "content": response.choices[0].message.content.strip()
            }

        except Exception as e:
            return {
                "func_type": "vlm",
                "content": f"Error during visual perception: {e}"
            }


if __name__ == "__main__":
    specification_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\xArm_specification.pdf"
    params_file = "C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\initialization\\robots\\specification\\params_general.json"

    roboticarmfunction = RoboticArmFunctions(specification_file, params_file)
    #roboticarmfunction.perform_assembly_tasks("human will do the housing step, you do the rest of the cable shark assembly process.")
    roboticarmfunction.perform_assembly_tasks("Execute the cable shark assembly process.")