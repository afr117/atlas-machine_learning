#!/usr/bin/env python3

import importlib.util
import sys

def import_neuron():
    spec = importlib.util.spec_from_file_location("Neuron", "4-neuron.py")
    neuron_module = importlib.util.module_from_spec(spec)
    sys.modules["4-neuron"] = neuron_module
    spec.loader.exec_module(neuron_module)
    return neuron_module

# Import Neuron class from dynamically loaded module
neuron_module = import_neuron()
Neuron = neuron_module.Neuron
