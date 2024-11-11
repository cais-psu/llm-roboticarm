# 🏭 llm-roboticarm
A setup for using LLMs in robotic arm assembly.

### Requirements
1. Install **Poetry** for dependency management.
   
2. Configure Poetry to use a local virtual environment in the project directory.

3. Install all project dependencies using Poetry. 

   * If installation encounters issues, first try locking dependencies, then proceed with installation.

4. **Python Version**: Use **Python 3.11.6**, as other libraries in this project rely on this specific version.

### Environment Variables
1. Set up an environment variable for your OpenAI API key, following OpenAI’s best practices for API key safety.

2. Ensure `tcl` is installed and configured correctly for your Python installation, as it may be required for certain interface functionalities.

### Running the Example
Use Poetry to run `llm_roboticarm_main.py` as the main file for the example setup.

## Features and Usage

1. **Voice Commands**: 
   - **Start** an instruction sequence by saying `"hello xarm"`.
   - **End** the command sequence by saying `"end of command"` when finished.

2. **Flexible Assembly Steps**:
   - Enable flexible command processing by setting `self.adaptation = True` in `robotic_arm_assembly.py`. This allows starting from specific steps (e.g., `"wedge"` assembly) by dynamically extracting information from the SOP. Note that the accuracy in step adaptation is a work in progress.

3. **Assembly Process Queries**:
   - Ask questions related to the assembly process to extract relevant information from `xArm_SOP.pdf`.

4. **Camera Setup**:
   - The Intel RealSense camera is mounted on the robotic gripper using a 3D-printed stand
   - [xArm Camera Stand](https://www.robotshop.com/products/xarm-camera-stand?srsltid=AfmBOoq05ORPif0tKjeGXJeipwM5LeSIpxAcENP_DbijBu2GvJISE1T8).

5. **IDE and Project Structure**:
   - Recommended IDE: **Visual Studio Code (VS Code)**.
   - Set the main directory to `llm-roboticarm`, not within nested directories.

6. **Dependency Management**:
   - Use Poetry for consistent and isolated dependency management.

7. **Operating System**:
   - Recommended OS: **Windows**.

## Developer Notes
- This project uses **Poetry** for dependency management and includes a `pyproject.toml` configuration for streamlined development.
