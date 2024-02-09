from openai_agent import (
    LlmAgent,
    User,
)
import utils

def create_user():
    user = User()
    return user

def create_robot_agents(robot_init_list, robot_functions):
    agents = []
    for robot_init in robot_init_list:
        config = utils.load_json_data(robot_init)
        try:
            for name, robot_data in config.items():
                functions_ = []
                # Check if 'functions' key exists and is a list
                if 'functions' in robot_data and isinstance(robot_data['functions'], list):
                    for function_name in robot_data['functions']:
                   # Retrieve each function by name using getattr
                        function_ref = getattr(robot_functions, function_name, None)
                        print(function_ref)
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
                agents.append(agent)
        except AttributeError as e:
            print(f"Error: {e} for agent {name}.")

    return agents