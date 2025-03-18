import os
import sys
import yaml
import logging
import importlib
import subprocess
from typing import List, Set

sys.path.append('/home/fboehning/fboehning')
from labtools import *
from labtools.labhandler import *
from labtools.logging_utils import *
from labtools.directory_utils import *
from labtools.experiment_utils import *

def run_experiment(experiment_config: dict) -> None:
    logging.debug(f"Running experiment")
    experiment=init_from_labh_dict(filter_dict_keylist(experiment_config, LABH_GLOBAL_KEYS))
    experiment.run()
    return

def main(multirun_dict: dict) -> None:

    _i=len(multirun_dict['experiment_paths'])
    for i, experiment_path in enumerate(multirun_dict['experiment_paths']):
        experiment_path=Path(experiment_path)
        experiment_config=get_yaml(experiment_path/"config.yaml")

        base_module_path=get_base_paths(get_dict_values(experiment_config, 'module_path'))[0]

        if experiment_config['done']:
            print(f"{i+1}/{_i}\t{experiment_path.name} already completed.")
            continue
        
        print(f"RUNNING {i+1}/{_i}\t{experiment_path.name}")

        stdout_log = os.path.join(experiment_path, f"stdout.log")
        stderr_log = os.path.join(experiment_path, f"stderr.log")

        with open(stdout_log, "w") as stdout_file, open(stderr_log, "w") as stderr_file:

            process = subprocess.Popen(
                ["python", __file__, experiment_path/"config.yaml"],
                stdout=stdout_file,
                stderr=stderr_file,
                env={
                    **os.environ,
                    'PYTHONPATH': base_module_path,
                },
            )
            process.wait()  # Wait for the experiment to finish


        if process.returncode == 0:
            print(f"EXPERIMENT {i+1}/{_i}\t{experiment_path.name} completed successfully.")
        else:
            print(f"EXPERIMENT {i+1}/{_i}\t{experiment_path.name} failed.")

    return

if __name__ == "__main__":
    # Access the input arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_argument>")
        sys.exit(1)

    LOGGING_FORMAT='%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT)
    
    input_path=Path(sys.argv[1])
    if not input_path.is_file():
        print(f"File not found: {input_path}")
        sys.exit(1)
    

    input_yaml=get_yaml(input_path)

    if 'experiment_paths' in input_yaml:
        print("START MULTIRUN")
        main(input_yaml)

    elif is_labh_dict(input_yaml):
        os.chdir(input_path.parent)
        print(f"Running single experiment {os.getcwd()=}")

        all_module_paths=list(set(get_dict_values(input_yaml, 'module_path')))
        reimport(all_module_paths)
        set_info(all_module_paths)

        run_experiment(input_yaml)
