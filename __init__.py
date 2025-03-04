import os
import logging
import warnings
import importlib

import pandas as pd

from types import MethodType
from functools import partial
from typing import Optional, Union

from labtools.directory_utils import *
from labtools.directory_utils import find_path_to

from labtools.dictionary_utils import *


LAB_SUFFIX='_lab'

def is_labhandler_dict(the_dict):
    if not isinstance(the_dict, dict): return False

    required_keys=['id','label','lab_name','class_parts', 'module_path']
    if all([key in the_dict for key in required_keys]):
        return True
    return False

def is_labhandler_object(the_object):
    if not hasattr(the_object, '__dict__'): return False
    return is_labhandler_dict(the_object.__dict__)

def is_labhandler(ref):
    if is_labhandler_dict(ref): return True
    if is_labhandler_object(ref): return True
    return False

def is_datahandle(ref):

    if is_labhandler_dict(ref) and ('Datahandle' in ref['class_parts']): return True
    elif is_labhandler_object(ref) and ('Datahandle' in ref.__dict__['class_parts']): return True
    return False

def init_from_dict(config_dict):
    assert is_labhandler_dict(config_dict), f"Invalid config_dict: {config_dict}"

    class_name=config_dict['class_parts'][-1]
    if class_name not in globals(): 
        import_class(config_dict['module_path'], class_name)
    
    the_class=globals()[class_name]
    class_obj=the_class(config_dict)

    print(f"{config_dict=}")
    if 'Datahandle' in config_dict.get('class_parts',[]):
        return class_obj.df
    else:
        class_obj

def init_from_object(obj):
    assert is_labhandler_object(obj), f"Invalid object: {obj}"
    return obj

def init_from_ref(ref):
    obj=None
    if is_labhandler_dict(ref):
        obj=init_from_dict(ref)
    
    elif is_labhandler_object(ref):
        obj=init_from_object(ref)

    if is_datahandle(ref):
        return obj.df
    
    return obj

def get_class_parts(the_class):
    if not isinstance(the_class, type): the_class=the_class.__class__
    class_parts=[c.__name__ for c in the_class.mro()[:-1]][::-1]
    return class_parts

def import_class(module_path: str, class_name: str):
    """Dynamically imports a class and defines it in the current namespace."""
    module = importlib.import_module(module_path)  # Import module
    the_class = getattr(module, class_name)  # Get the_class from module
    globals()[class_name] = the_class  # Define the_class in global namespace

class Labhandler(object):
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
    
    def load_from_object(self, the_object:object) -> None:
        self.class_parts=get_class_parts(the_object)
        self.module_path=getattr(the_object,'__module__', None)
    
    def load_from_dict(self, the_dict: dict) -> None:
        if not is_labhandler_dict(the_dict): return 

        self.id=the_dict.pop('id')
        self.class_parts=the_dict.pop('class_parts')
        self.module_path=the_dict.pop('module_path')
        self.update_config(the_dict)

    def load_from_path(self, path: Union[str, Path]) -> None:
        the_dict = get_yaml(Path(path)/'config.yaml')
        self.load_from_dict(the_dict)
    
    def load(self, ref=None, obj=None, label:str=None, **kwargs) -> None:

        if obj is not None: self.load_from_object(obj)

        if isinstance(ref, (str, Path)):
            self.load_from_path(ref)

        elif is_labhandler_dict(ref): 
            self.load_from_dict(ref)

        if isinstance(ref, (int, type(None))):

            os.makedirs(self._class_path, exist_ok=True)
            
            if ref == -1:
                self.id=get_max_id(self._class_path)

            elif isinstance(ref, int): self.id=ref
            else: self.id=get_max_id(self._class_path)+1

        if self.is_spawned: #update label if it was renamed
            fn=get_filename_from_id(self._class_path, self.id, warn=True)
            label=label or re.match(ID_PATTERN, fn).group(2)

        self.label = label or getattr(self, 'label', None) or self.class_parts[-1]
        self._name=f"{self.id:04d}_{self.label}"

        if self.is_spawned:
            config_dict = get_yaml(f"{self._path}/config.yaml")
            config_dict = update_dict(config_dict, self.config, overwrite_if_conflict=True, interpret_none_as_val=True)
            self.update_config(config_dict)
            self.spawn_config()

    def __init__(self, ref=None, obj=None, lab_path:str=None, **kwargs):
        self.load_lab(lab_path)

        self._config_key_order = ['id','label', 'lab_name','class_parts','module_path','path']
        self._invoke_keys=[]
        self._attach_methods = ['get_overview','get_config', 'spawn', 'spawn_config','update_config','config','is_spawned','_path','_class_path']
        self._config_exclude_keys = ['labh','class_path','logger','df','model']+self._attach_methods

        if ref is not None:
            self.load(ref, obj, **kwargs)

    def get_overview(self, keypaths:list=[]) -> pd.DataFrame:

        matching_paths=find_matching_paths(target_pattern=ID_PATTERN, base_path=self._class_path)

        overview_data=[]
        for path in matching_paths:
            labh=Labhandler(path)

            overview_row=dict()
            overview_row['class_parts']=labh.class_parts
            overview_row['id']=labh.id
            overview_row['label']=labh.label

            config_dict=labh.config
            for keypath in keypaths:

                value=None
                keypath_list=split_keypath(keypath)

                if keypath in config_dict.keys():
                    value=config_dict[keypath]

                elif len(keypath_list)==1:
                    ambiguous_keypaths = get_dict_keypaths(config_dict, keypath_list[0])

                    if len(ambiguous_keypaths)==1:
                        value=get_dict_value_from_keypath(config_dict, ambiguous_keypaths[0])

                    elif len(ambiguous_keypaths)>1:

                        ambiguous_values=[]
                        for ambiguous_keypath in ambiguous_keypaths:
                            ambiguous_values.append(get_dict_value_from_keypath(config_dict, ambiguous_keypath))
                        
                        if len(set(ambiguous_values))==1:
                            value=ambiguous_values[0]
                        else:
                            logging.warning(f"Multiple values found for '{keypath}' in config_dict: {ambiguous_values}")

                elif has_keypath(config_dict, keypath_list):
                    value=get_dict_value_from_keypath(config_dict, keypath_list)


                if value is not None:
                    for i in range(1,len(keypath_list)+1):
                        keypath_str='.'.join(keypath_list[len(keypath_list)-i:])
                        if keypath_str not in overview_row.keys():
                            break
                    
                    overview_row[keypath_str]=value

            overview_data.append(overview_row)

        return pd.DataFrame(overview_data)

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
        further_items=filter_dict_keypatterns(further_items, [r'^_'], invert=True)

        for key in self._invoke_keys:
            if key not in further_items: continue
            val=further_items[key]
            val=invoke_dict_callable(val, 'get_config')
            val=invoke_dict_callable(val, '__dict__')
            further_items[key]=val

        further_items=filter_dict_valuetypes(further_items,valuetypes=[str,int,float,bool,dict,list,tuple,set,type(None)])
        further_items=filter_dict_values(further_items, [None], invert=True)

        config_dict.update(further_items)
        return config_dict

    @property
    def config(self):
        return self.get_config()

    @property
    def class_path(self):
        return (Path(self.lab_name)/Path(*self.class_parts)).as_posix()

    @property
    def _class_path(self):
        return (Path(self._root_path)/Path(self.lab_name)/Path(*self.class_parts)).as_posix()

    @property
    def _path(self):

        fn = get_filename_from_id(self._class_path, self.id, warn=False)
        fn = fn or self._name

        return (Path(self._class_path)/fn).as_posix()

    @property
    def is_spawned(self):
        fn=get_filename_from_id(self._class_path, self.id, warn=False)
        return bool(fn)
        #return os.path.exists(f'{self._path}')

    def spawn_config(self):
        os.makedirs(self._path, exist_ok=True)
        set_yaml(self.config,f"{self._path}/config.yaml")

    def spawn(self):
        self.spawn_config()

    def get_initialized_objects(self, locals, var_names:list = None) -> list:
        """
        Tries to initialize lab objects and returns them as list.

        Args:
            locals: Dictionary containing the local variables of the calling function.
            var_names: List of variable names to be initialized.
        
        Returns:
            List of initialized objects.
        """

        init_objs=[None]*len(var_names)

        for i,var_name in enumerate(var_names):
            init_objs[i] = self.config.get(var_name, None) or locals.get(var_name, None)

            if isinstance(init_objs[i], list) and all([is_labhandler(v) for v in init_objs[i]]):
                init_objs[i]=[init_from_ref(v) for v in init_objs[i]]

            elif is_labhandler(init_objs[i]):
                init_objs[i]=init_from_ref(init_objs[i])

            # else:
            #     warnings.warn(f"{var_name}:{init_objs[i]} is no valid labhandler.")
            
            self.logger.debug(f"{var_name}:{init_objs[i]}")

        return init_objs
    
    def attach(self, locals: dict, var_names:list = None, **kwargs):
        """
        Attach lab attributes to target object.
        locals: dict
            Dictionary containing the local variables of the calling function.
        var_names: list
            List of keys to be preloaded from locals.
        kwargs: dict
        """

        del locals['labh']
        obj=locals.pop('self', None)
        ref=locals.pop('ref', None)
        locals_kwargs=locals.pop('kwargs', None)
        #search for label in locals, kwargs, and kwargs['kwargs']
        label=(locals.pop('label', None)) or (locals_kwargs.get('label', None)) or (kwargs.get('label', None))

        self._invoke_keys=list(locals.keys()) # keys to be invoked in get_config/__dict__ -> better for displaying in yaml
        self.load(ref, obj, label)

        #Attach Locals to target object
        for k,v in locals.items(): setattr(obj, k, v)

        #Attach Lab Attributes to target object
        for k,v in vars(self).items():
            if (k in locals): continue
            setattr(obj, k, v)
        setattr(obj, 'logger', self.logger)

        for method_name in self._attach_methods:

            if hasattr(obj, method_name): continue #method already exists --> do not overwrite it!

            method = getattr(self.__class__, method_name, None)

            if isinstance(method, property):
                setattr(obj.__class__, method_name, method)

            if callable(method):
                setattr(obj, method_name, MethodType(method, obj))

        if var_names is not None:
            return self.get_initialized_objects(locals, var_names)
        return
  





            
        














