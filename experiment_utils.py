import os
import sys
import yaml
import logging
import importlib
import pandas as pd
from typing import List, Set, Union

import labtools
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

def generate_multirun(the_input: Union[List[str], pd.DataFrame]) -> dict:

    if isinstance(the_input, pd.DataFrame):
        if 'done' in the_input.columns:
            the_input=the_input[the_input['done']==False]

        if '_path' in the_input.columns:
            the_input=the_input['_path'].to_list()

    labtools_dir=Path(labtools.__path__[0]).parent
    runner_path=Path(labtools.__path__[0])/'experiment_runner.py'

    multirun_path=f"{os.path.commonpath(the_input)}/multirun.yaml"
    multirun_yaml=dict(

        experiment_paths=[f"{path}" for path in the_input]
    )

    set_yaml(multirun_yaml,multirun_path)

    print()
    print("Run the following command to start the multirun:")
    print()
    print(f"\tconda activate beesup; python {runner_path} {multirun_path}")

    return multirun_yaml

