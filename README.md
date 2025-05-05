# 🏭 llm-roboticarm
A setup leveraging LLMs for robotic arm assembly.

---

### Requirements
1. **Poetry**: Install Poetry for dependency management.
2. **Python Version**: Use **Python 3.11.6** (required by project libraries).
3. **Environment Variable**: Set up an environment variable for your OpenAI API key following OpenAI's best practices.
4. **Tcl Configuration**: Ensure Tcl is installed and configured correctly for Python.

---

### Installation
1. **Configure Poetry**:
   - pip install poetry
   - Use a local virtual environment within the project directory:
      poetry config virtualenvs.in-project true
      poetry install

   - If installation issues arise, try:
      poetry lock
      poetry install

2. **Run the Example**:
   - Execute `llm_roboticarm_main.py` using Poetry as the main example file.

---

## Features and Usage

1. **Voice Commands**: 
   - **Start** commands with `"hello xarm"`.
   - **End** commands with `"end of command"` when finished.

2. **Flexible Assembly Steps**:
   - To allow commands to start from specific steps (e.g., `"wedge"` assembly), set `self.adaptation = True` in `robotic_arm_assembly.py`. The system will dynamically use information from the SOP, although accuracy is still being refined.

3. **Assembly Queries**:
   - Ask questions about the assembly process to extract information from `xArm_SOP.pdf`.

4. **Camera Setup**:
   - The Intel RealSense camera is mounted on the gripper with a 3D-printed stand:
     - [xArm Camera Stand](https://www.robotshop.com/products/xarm-camera-stand?srsltid=AfmBOoq05ORPif0tKjeGXJeipwM5LeSIpxAcENP_DbijBu2GvJISE1T8)

5. **IDE and Project Structure**:
   - **Recommended IDE**: Visual Studio Code (VS Code)
   - **Project Directory**: Set the main directory to `llm-roboticarm`.

6. **Dependency Management**:
   - Use **Poetry** for isolated and consistent dependency handling.

7. **Operating System**:
   - **Recommended OS**: Windows

8. **Robot Movement Configuration**:
   - `params_movement.json` holds xArm movement coordinates. Update this file to adjust to specific xArm setups or calibrate based on the model in use for optimal alignment.

9. **Using urx library for UR5e robot arm**
   - urx=0.11.0 version
   - in robot.py, below change needs to be made:
       def getl(self, wait=False, _log=True):
        """
        return current transformation from tcp to current csys
        """
        t = self.get_pose(wait, _log)
        return t.pose_vector.get_array().tolist()
   - in rurobot.py, below change is required:
       def movex(self, command, tpose, acc=0.01, vel=0.01, wait=True, relative=False, threshold=None):
        """
        Send a move command to the robot. Since UR robots have several methods, this sends
        whatever is defined in 'command' string (e.g., "movel", "movep").
        """
        # Convert PoseVector to array if needed
        if hasattr(tpose, "array"):
            tpose = tpose.array

        # Handle relative motion
        if relative:
            l = self.getl()
            tpose = [v + l[i] for i, v in enumerate(tpose)]

        # Format and send the URScript command
        prog = self._format_move(command, tpose, acc, vel, prefix="p")
        self.send_program(prog)

        # Optionally wait until the motion is complete
        if wait:
            self._wait_for_move(tpose[:6], threshold=threshold)
            return self.getl()

---

## Developer Notes
- **Dependency Management**: This project uses **Poetry** and includes a `pyproject.toml` configuration for streamlined development.
