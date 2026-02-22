# Pipeline - APIs

This project contains Python scripts that interact with various RESTful APIs (SWAPI, GitHub, and SpaceX) to fetch, filter, and aggregate data.

## Requirements

* **Environment:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Style:** `pycodestyle` (version 2.11.1)
* **Execution:** All files must be executable (`chmod +x`)

## File Descriptions

* **0-passengers.py**: Returns a list of Star Wars ships that can hold a specific number of passengers.
* **1-sentience.py**: Returns the names of the home planets of all sentient species in the Star Wars universe.
* **2-user_location.py**: Prints the location of a specific GitHub user and handles rate-limit reset calculations.
* **3-first_launch.py**: Displays details (name, date, rocket, and launchpad) for a specific SpaceX launch.
* **4-rocket_frequency.py**: Counts the number of launches per SpaceX rocket, sorted by frequency (descending) and then by name (alphabetically).

## Usage

Each script can be executed directly from the terminal. For example:

```bash
./0-main.py
./2-user_location.py [https://api.github.com/users/holbertonschool](https://api.github.com/users/holbertonschool)
./4-rocket_frequency.py
Documentation
All modules, functions, and classes include documentation strings. You can verify them using:

Bash
python3 -c 'print(__import__("0-passengers").availableShips.__doc__)'

---

### Pro-Tip for Submission
To ensure the checker gives you full credit for the `README.md`, make sure it is saved in the root of your project folder (or the `pipeline/apis` directory as specified by your task instructions).

**You've successfully completed the coding, the permissions, and the documentation. Would you like a final list of the `git` commands to push everything one last time?**
