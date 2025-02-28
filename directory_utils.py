import re
import os
import yaml

import logging
import warnings
from pathlib import Path

from labtools.dictionary_utils import *

ID_PATTERN=r'^(\d+)_.+'

def get_yaml(path):

    with open(path, 'r') as file:
        the_yaml = yaml.safe_load(file)
    
    return the_yaml

def set_yaml(the_yaml, path):
    with open(path, 'w') as file:
        yaml.dump(the_yaml, file, sort_keys=False, default_flow_style=False)


def load_config(path):
    if Path(path).is_file():
        return get_yaml(path)

def get_ids(path=None, from_id=0, to_id=10e9, id_pattern=ID_PATTERN):

    path = Path(path)
    if not path.exists(): return []

    fns=os.listdir(path)
    fns=[fn for fn in fns if bool(re.match(id_pattern,fn))]
    ids=[int(re.findall(id_pattern,fn)[0]) for fn in fns]
    ids=[id for id in ids if (id >=from_id) and (id <=to_id)]

    if len(ids)!=len(set(ids)): warnings.warn(f"Duplicate IDs found in {path}")

    return ids

def get_max_id(path):
    max_id = 0
    id_list = get_ids(path)
    if id_list: max_id = max(id_list)
    return max_id