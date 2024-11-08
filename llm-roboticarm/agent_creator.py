from llm_agent import LlmAgent, User
import general_utils

def create_user():
    """
    Creates and returns a new instance of a User.

    Returns:
        User: A new User object instance.
    """
    user = User()
    return user

def create_robot_agents(robot_init_list, robot_functions):
    """
    Creates a list of LLM-based robot agents based on provided initialization data
    and function references. Each agent is configured with specific functions and 
    instructions as defined in its configuration file.

    Args:
        robot_init_list (list): A list of file paths to JSON configuration files for each robot.
        robot_functions (module): A module containing function references to be assigned to each agent.

    Returns:
        list: A list of LlmAgent objects configured as robot agents.
    
    Raises:
        AttributeError: If a function specified in the configuration does not exist in robot_functions.
    """
    agents = []

    for robot_init in robot_init_list:
        # Load configuration data from JSON file specified in robot_init
        config = general_utils.load_json_data(robot_init)
        try:
            for name, robot_data in config.items():
                functions_ = []

                # Check if 'functions' key exists in robot_data and is a list
                if 'functions' in robot_data and isinstance(robot_data['functions'], list):
                    for function_name in robot_data['functions']:
                        function_ref = getattr(robot_functions, function_name, None)
                        if function_ref is not None:
                            functions_.append(function_ref)
                        else:
                            print(f"Function '{function_name}' not found in functions module for agent {name}.")

                agent = LlmAgent(
                    model="gpt-4o",
                    name=name, 
                    annotation=robot_data.get('annotation', None), 
                    instructions=robot_data.get('instructions', None),
                    functions_ = functions_,
                )
                agents.append(agent) 
        except AttributeError as e:
            print(f"Error: {e} for agent {name}.")

    return agents
