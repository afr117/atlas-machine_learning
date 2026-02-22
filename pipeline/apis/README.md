\# Pipeline - APIs



\## Description

This project focuses on interacting with external APIs using Python. It covers handling HTTP requests, parsing JSON data, and managing pagination to retrieve complete datasets from RESTful services.



\## Requirements

\* \*\*Editors:\*\* `vi`, `vim`, `emacs`

\* \*\*Environment:\*\* Ubuntu 20.04 LTS

\* \*\*Language:\*\* Python 3.9

\* \*\*Style Guide:\*\* `pycodestyle` (version 2.11.1)



\## Tasks



\### 0. Can I join?

A script that queries the \[SWAPI API](https://swapi-api.hbtn.io/) to find starships capable of carrying a specific number of passengers.



\* \*\*File:\*\* `0-passengers.py`

\* \*\*Prototype:\*\* `def availableShips(passengerCount):`

\* \*\*Functionality:\*\* \* Fetches all starships using pagination.

&nbsp;   \* Parses passenger counts (handling strings with commas).

&nbsp;   \* Returns a list of ship names that meet or exceed the `passengerCount`.



\## Usage

To test the script, use the provided main file:



```bash

./0-main.py

