import os
import logging
import warnings
import importlib

from types import MethodType
from functools import partial
from typing import Optional, Union

from labtools.directory_utils import *
from labtools.dictionary_utils import *

LAB_SUFFIX='_lab'

#def init_from_model(model):



def is_lab_config(config_dict):
    if not isinstance(config_dict, dict): return False

    required_keys=['id','lab_name','class_parts', 'module_path']
    if all([key in config_dict for key in required_keys]):
        return True
    return False

def is_lab_object(obj):
    if not hasattr(obj, '__dict__'): return False
    return is_lab_config(obj.__dict__)

def is_lab_ref(ref):
    if isinstance(ref, dict): return is_lab_config(ref)
    else: return is_lab_object(ref)
    return False



def init_from_config(config_dict):
    assert is_lab_config(config_dict), f"Invalid config_dict: {config_dict}"

    class_name=config_dict['class_parts'][-1]
    if class_name not in globals(): 
        import_class(config_dict['module_path'], class_name)
    
    the_class=globals()[class_name]
    class_obj=the_class(config_dict)

    print(f"{config_dict=}")

    if 'DataHandle' in config_dict['class_parts']:
        return class_obj.df
    else:
        class_obj

def init_from_object(obj):
    assert is_lab_object(obj), f"Invalid object: {obj}"
    return obj

def init_from_ref(ref):
    assert is_lab_ref(ref), f"Invalid ref: {ref}"
    if isinstance(ref, dict): return init_from_config(ref)
    else: return init_from_object(ref)



def get_class_parts(the_class):
    if not isinstance(the_class, type): the_class=the_class.__class__
    class_parts=[c.__name__ for c in the_class.mro()[:-1]][::-1]
    return class_parts

def import_class(module_path: str, class_name: str):
    """Dynamically imports a class and defines it in the current namespace."""
    module = importlib.import_module(module_path)  # Import module
    the_class = getattr(module, class_name)  # Get the_class from module
    globals()[class_name] = the_class  # Define the_class in global namespace

class LabHandler(object):
    logger = logging.getLogger(__name__)

    def load_lab(self, path:str=None) -> None:
        if path is None: path=os.getcwd()
        
        path_parts = []
        for path_part in Path(path).parts:
            path_parts.append(path_part)
            current_path = Path(*path_parts)

            if current_path.is_dir() and path_part.endswith(LAB_SUFFIX):
                self._root_path = Path(*path_parts[:-1]).as_posix() # _ denotes absolute paths/dirs
                self.lab_name = path_part
                continue
        return
    
    def load_ref(self, ref=None, obj=None) -> None:

        self.class_parts=get_class_parts(obj)
        self.module_path=obj.__module__

        class_path=(Path(self.lab_name)/Path(*self.class_parts)).as_posix()
        _class_path=(Path(self._root_path)/Path(class_path)).as_posix()
        os.makedirs(_class_path, exist_ok=True)

        if isinstance(ref, dict):
            self.update_config(ref)
        
        elif isinstance(ref, (int, type(None))):

            if ref == -1:
                self.id=get_max_id(_class_path)

            elif isinstance(ref, int): self.id=ref

            else: self.id=get_max_id(_class_path)+1

            self.name=f"{self.id:04d}_{self.class_parts[-1]}"
            self.path=(Path(class_path)/Path(self.name)).as_posix()
        
        self._path=(Path(_class_path)/Path(self.name)).as_posix()

        if self.is_spawned:
            self.logger.debug(f"Loading {self.name} from {self._path}")
            config_path=(Path(self._path)/Path('config.yaml'))

            if config_path.exists():
                config_dict=get_yaml(config_path)
                self.update_config(config_dict)
            else:
                warnings.warn(f"config.yaml not found in {self._path}")

        else:
            self.logger.debug(f"Creating {self.name} at {self._path}")
            #os.makedirs(self._path, exist_ok=True)
        
    def __init__(self, obj=None, ref=None, lab_path:str=None):
        self.load_lab(lab_path)

        self._config_key_order = ['id','name', 'lab_name','class_parts','module_path','path']
        self._attach_methods = ['get_config', 'spawn', 'spawn_config','update_config']
        self._config_exclude_keys = ['lab']+self._attach_methods

        if obj is not None:	self.load_ref(ref, obj)
    
    def update_config(self, mixin_dict, overwrite_if_conflict=True, interpret_none_as_val=True):
        
        #self.logger.debug(f"mixin_dict: {mixin_dict}")

        updated_config_dict = update_dict(
            self.get_config(),
            mixin_dict,
            overwrite_if_conflict=overwrite_if_conflict,
            interpret_none_as_val=interpret_none_as_val
            )
        
        #self.logger.debug(f"updated_config_dict: {mixin_dict}")
        for k, v in updated_config_dict.items(): setattr(self, k, v)
        return

    def get_config(self):
        config_dict=dict()

        for k in self._config_key_order:
            if k in config_dict: continue
            if hasattr(self, k): config_dict[k]=getattr(self, k)

        further_items=self.__dict__
        further_items=filter_dict_keylist(further_items, self._config_exclude_keys, invert=True)
        further_items=invoke_dict_callable(further_items, 'get_config')
        further_items=invoke_dict_callable(further_items, '__dict__')
        further_items=filter_dict_valuetypes(further_items,valuetypes=[str,int,float,bool,dict,list,tuple,set,type(None)])
        further_items=filter_dict_keypatterns(further_items, [r'^_'], invert=True)
        config_dict.update(further_items)

        return config_dict

    @property
    def config(self):
        return self.get_config()
    
    @property
    def is_spawned(self):
        return os.path.exists(f'{self._path}')

    def spawn_config(self):

        os.makedirs(self._path, exist_ok=True)
        config_dict=self.get_config()
        _config_path=(Path(self._path)/Path('config.yaml')).as_posix()
        set_yaml(config_dict, _config_path)
        
    def spawn(self):
        self.spawn_config()

    def get_init_objects(self, locals, keys:list = None) -> list:
        if keys is None: return 

        init_objects=[None]*len(keys)
        for i,key in enumerate(keys):
            
            if not key in locals: 
                warnings.warn(f"{key} not found in locals.")
                continue
            init_objects[i]=locals[key]

            if key not in self.config:
                if self.is_spawned:
                    warnings.warn(f"{key} not found in config.")
                continue
            init_objects[i]=self.config[key]

            if isinstance(init_objects[i], list) and all([is_lab_ref(v) for v in init_objects[i]]):
                init_objects[i]=[init_from_ref(v) for v in init_objects[i]]
            elif is_lab_ref(init_objects[i]):
                init_objects[i]=init_from_ref(init_objects[i])
            else: 
                warnings.warn(f"{key}:{init_objects[i]} is no valid lab config.")

        return init_objects
    
    def attach(self, locals, preinit_keys:list = None, **kwargs):
        """
        Attach lab attributes to target object.
        locals: dict
            Dictionary containing the local variables of the calling function.
        preinit_keys: list
            List of keys to be preloaded from locals.
        kwargs: dict
        """

        del locals['lab_handler']
        obj=locals.pop('self', None)
        ref=locals.pop('ref', None)

        self.load_ref(ref, obj)

        #Attach Lab Attributes to target object
        for k,v in vars(self).items(): setattr(obj, k, v)

        for method_name in self._attach_methods:

            if hasattr(obj, method_name): continue #method already exists --> do not overwrite it!

            method = getattr(self.__class__, method_name, None)
            if callable(method):
                setattr(obj, method_name, MethodType(method, obj))

        if preinit_keys is not None:
            return self.get_init_objects(locals, preinit_keys)
        
        else:
            return
  





            
        














