import os
import sys
import yaml
import logging
import importlib

from labtools.directory_utils import *

def get_base_paths(module_paths:List[str]) -> List[str]:
    base_paths = set()

    for module_name in set(module_paths):
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)
            if hasattr(module, '__file__'):
                module_file = os.path.abspath(module.__file__)
                
                # Remove the module path to find the base directory
                base_path = module_file
                for _ in module_name.split('.'):
                    base_path = os.path.dirname(base_path)
                
                base_paths.add(base_path)
        except ImportError as e:
            print(f"[ERROR] Could not import {module_name}: {e}")

    return list(base_paths)

