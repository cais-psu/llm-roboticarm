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
* Formatting with [black](https://github.com/psf/black)
* Linting with [ruff](https://github.com/astral-sh/ruff)
