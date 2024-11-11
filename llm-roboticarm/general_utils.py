import json
import os
import hashlib

def load_json_data(file_path):
    """
    Loads JSON data from a specified file path.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict: The JSON data as a Python dictionary.
    """    
    with open(file_path, 'r') as file:
        return json.load(file)

# Get the paths of all JSON files in the specified directory.
def get_init_files(directory_path):
    """
    Retrieves a list of paths for all JSON files in a specified directory.

    Args:
        directory_path (str): The path to the directory to search for JSON files.

    Returns:
        list: A list of file paths for each JSON file in the directory.
    """    
    json_file_paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            json_file_paths.append(file_path)
    return json_file_paths

def load_specification(file_path):
    """
    Loads a specification file as text from a given path relative to the current directory.

    Args:
        file_path (str): The path to the specification file.

    Returns:
        str: The content of the specification file, or an error message if not found.
    """    
    try:
        current_directory = os.getcwd()
        spec_path = current_directory + file_path
        with open(spec_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."

def get_inbox_identifier(sender, content):
    """
    Generates a unique inbox identifier based on the sender and content.

    Args:
        sender (str): The identifier of the sender.
        content (str): The content for which the identifier is generated.

    Returns:
        str: A unique identifier combining sender and a hash of the content.
    """    
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    inbox_identifier = f"{sender}_{content_hash}"

    return inbox_identifier 

def get_command_identifier(command):
    """
    Generates a unique command identifier based on the command content.

    Args:
        command (str): The command for which the identifier is generated.

    Returns:
        str: An 8-character hash-based identifier for the command.
    """    
    content_hash = hashlib.md5(command.encode()).hexdigest()[:8]
    command_identifier = f"{content_hash}"

    return command_identifier 
