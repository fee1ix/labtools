import re
import os
import yaml

import logging
import warnings
from pathlib import Path

from labtools.dictionary_utils import *

ID_PATTERN=r'^(\d+)_(.+)'

def get_yaml(path):

    with open(path, 'r') as file:
        the_yaml = yaml.safe_load(file)
    
    return the_yaml

def set_yaml(the_yaml, path):
    with open(path, 'w') as file:
        yaml.dump(the_yaml, file, sort_keys=False, default_flow_style=False)

def get_filename_from_id(path, id: int, id_pattern=ID_PATTERN, warn=True):
    path = Path(path)
    if not path.exists(): return None
    
    fns = os.listdir(path)
    fns = [fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids = [int(re.match(id_pattern,fn).group(1)) for fn in fns]

    if ids.count(id)==1:
        return fns[ids.index(id)]

    elif ids.count(id)==0:
        if warn: warnings.warn(f"ID {id} not found in {path}")
    
    elif ids.count(id)>1:
        if warn: warnings.warn(f"Multiple IDs {id} found in {path}")
    
    return None

def get_config(path=None, id:int=None) -> dict:
    if path is None: path=os.getcwd()
    path=Path(path)

    if path.is_dir():
        if id is not None:
            fn=get_filename_from_id(path, id)
            path = path/fn
    
    return get_yaml(path/'config.yaml')

def get_ids(path=None, from_id=0, to_id=10e9, id_pattern=ID_PATTERN):

    path = Path(path)
    if not path.exists(): return []

    fns = os.listdir(path)
    fns = [fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids = [int(re.match(id_pattern,fn).group(1)) for fn in fns]
    ids = [id for id in ids if (id >=from_id) and (id <=to_id)]

    if len(ids)!=len(set(ids)): warnings.warn(f"Duplicate IDs found in {path}")

    return ids

def get_max_id(path):
    max_id = 0
    id_list = get_ids(path)
    if id_list: max_id = max(id_list)
    return max_id

def find_path_to(target_name, base_path=os.getcwd()):
    for root, dirs, files in os.walk(base_path):
        if target_name in dirs+files:
            return os.path.join(root, target_name)
    return None

def find_matching_paths(target_pattern=ID_PATTERN, base_path:str=os.getcwd()):
    paths = []
    regex = re.compile(target_pattern)  # Compile the regex pattern

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs+files:
            if regex.search(dir_name):  # Check if directory name matches pattern
                paths.append(os.path.join(root, dir_name))

    return paths