# 🏭 llm-roboticarm
A setup for using LLMs in robotic arm assembly.

# Contents
* `LlmAgent` - an individual agent backed by the OpenAI model of your choice
* `FunctionAnalyzer` - Python introspection module for creating descriptions necessary for LLM callbacks
* `AgentManagementSystem` - manages the individual agents

# Setup
* [Set an environment variable](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) `OPENAI_API_KEY` for your OpenAI API key
* Initialize with [Poetry](https://python-poetry.org/docs/) `poetry install`
* Run example `poetry run python llm_roboticarm_main.py` 🚀


# Dev notes
* Dependency management with [poetry](https://github.com/python-poetry/poetry)


# Downloads
* https://github.com/oschwartz10612/poppler-windows/releases/#
* In the Edit Environment Variable window, click "New" and add the path to the bin 
* directory of your poppler installation (e.g., C:\poppler-22.04.0\bin).
* Click "OK" to close all the windows.
* For "unstructured_inference" library, do
* poetry shell
* pip install unstructured_inference, this should work.