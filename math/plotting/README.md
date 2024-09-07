# Atlas Machine Learning - Plotting Project

## Project Overview
This project focuses on creating various types of plots using Python's `matplotlib` and `numpy` libraries on Ubuntu 20.04 LTS. Each task is aimed at visualizing mathematical functions and datasets in different ways.

## Environment
- **Python version**: 3.9
- **Numpy version**: 1.25.2
- **Matplotlib version**: 3.8.3
- **Pycodestyle version**: 2.11.1
- **Operating System**: Ubuntu 20.04 LTS

## Prerequisites
Ensure you have the following packages installed:
- **Matplotlib**: `pip install --user matplotlib==3.8.3`
- **Pillow**: `pip install --user Pillow==10.2.0`
- **Python TK**: `sudo apt-get install python3-tk`

## X11 Forwarding Setup
To forward GUI applications (e.g., plots) to your local machine, configure your `Vagrantfile` as follows:
```ruby
Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end
For Mac Users:
Install XQuartz and restart your computer.
For Windows Users:
Follow the instructions for configuring X11 forwarding on Windows.
Current Task: Plotting Graphs
0. Line Graph
The goal is to plot a line graph of the function y = x^3, with the x-axis ranging from 0 to 10. The line should be solid and red.

File Structure:
0-line.py: Contains the function to generate and display the graph.
0-main.py: The entry point to run the graph plotting function.
How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/your_username/atlas-machine_learning.git
Navigate to the project directory:
bash
Copy code
cd math/plotting
Execute the main script:
bash

