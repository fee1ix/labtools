import os
import sys
import copy
import pickle
import logging
import warnings
import importlib

import pandas as pd

from types import MethodType
from functools import partial
from typing import Optional, Union

from labtools import *
from labtools.system_utils import TIMESTAMP_FORMAT, get_datetime

from labtools.directory_utils import *
from labtools.dictionary_utils import *

def get_labhandler():
    """Returns an instance of Labhandler if it exists in sys.modules, else None."""
    if 'labtools.labhandler' in sys.modules:
        module = sys.modules['labtools.labhandler']
        return getattr(module, 'Labhandler', None)() if hasattr(module, 'Labhandler') else None
    return None

LAB_SUFFIX='_lab'
LABH_FILE_KEYS=['type','filename']
LABH_LOCAL_KEYS=['class_parts','module_path']
LABH_GLOBAL_KEYS=['id','label','lab_name','class_parts','module_path']

def is_empty(x):

    if x is None: return True
    if isinstance(x, pd.DataFrame): return x.empty
    if isinstance(x, (list, tuple, set, dict)): return len(x)==0
    return x is None

def is_labh_dict(the_input, required_keys=LABH_GLOBAL_KEYS) -> bool:
    if not isinstance(the_input, dict): return False
    if all([key in the_input for key in required_keys]): return True
    return False

def is_labh_object(the_input, required_keys=LABH_GLOBAL_KEYS) -> bool:
    if not hasattr(the_input, '__dict__'): return False
    return is_labh_dict(the_input.__dict__, required_keys)

def is_labh_reference(the_input, required_keys=LABH_GLOBAL_KEYS) -> bool:

    if isinstance(the_input, dict): return is_labh_dict(the_input, required_keys)
    elif hasattr(the_input, '__dict__'): return is_labh_dict(the_input.__dict__, required_keys)

    return False

def is_labh_datahandle_reference(the_input):

    if is_labh_dict(the_input, LABH_LOCAL_KEYS) and ('Datahandle' in the_input['class_parts']): return True
    elif is_labh_object(the_input, LABH_LOCAL_KEYS) and ('Datahandle' in the_input.__dict__['class_parts']): return True
    return False


def init_from_labh_object(the_object, **kwargs):
    return the_object

def init_from_labh_dict(the_dict, **kwargs):
    assert is_labh_dict(the_dict, LABH_LOCAL_KEYS), f"{the_dict} is not a valid labhandler dictionary."
    class_name=the_dict['class_parts'][-1]

    # if class_name=='DataFrame':
    #     class_name=''

    if the_dict.get('class_parts', None) == ['FinetuningExperiment', 'InjectionExperiment']:
        print(f"{the_dict=}")

    if class_name not in globals():
        the_class = get_class(class_name,the_dict['module_path'])
        logging.debug(f"get {class_name} from {the_dict['module_path']} {id(the_class)=}")
    else: 
        the_class = globals()[class_name]
        logging.debug(f"use {class_name} from globals() as {the_class} {id(the_class)=}")
    
    if is_labh_dict(the_dict, LABH_GLOBAL_KEYS):
        logging.debug(f"its a global labhandler dict")
        the_dict['ref']=the_dict.pop('id', None)

    else:
        logging.debug(f"its no global labhandler local dict")

    logging.debug(f"{the_class=}")
    #logging.debug(f"{filter_dict_keylist(the_dict, LABH_GLOBAL_KEYS)=}")

    kwargs.update(**the_dict)
    if the_dict.get('class_parts', None) == ['FinetuningExperiment', 'InjectionExperiment']:
        print(f"{kwargs=}")

    the_object=the_class(**kwargs)

    # if the_dict.get('label', None) == 'RAG_CHUNKS':
    #     print(f"the_object.config={json.dumps(the_object.config, ensure_ascii=False, indent=2)}")
        


    return the_object

def init_from_labh_reference(the_input, **kwargs):
    """
    Initializes a labhandler attached object from a dictionary or an object reference.
    """
    the_object=the_input
    if is_labh_dict(the_input, LABH_LOCAL_KEYS):
        the_object=init_from_labh_dict(the_input, **kwargs)
    
    return the_object

def load_labh_file_from_dict(the_dict:dict, path:str=''):
    assert is_labh_dict(the_dict, LABH_FILE_KEYS), f"{the_dict} is not a valid labhandler file reference."

    if the_dict['type'].endswith('DataFrame'):
        return pd.read_pickle((Path(path)/the_dict['filename']).as_posix())
    else:
        raise NotImplementedError(f"Object of type {the_dict['type']} not supported.")

def save_labh_file(the_object, filename:str, path:str='', overwrite=False, **kwargs):
    save_path=(Path(path)/filename).as_posix()
    if os.path.exists(save_path) and not overwrite: return #do not overwrite existing files

    if isinstance(the_object, pd.DataFrame):
        the_object.to_pickle(save_path)
    else:
        raise NotImplementedError(f"Object of type {type(the_object)} not supported.")

def get_labh_dict_from_file(the_object:object, var_name:str, **kwargs):
    labh_dict=dict()
    labh_dict['type']=the_object.__class__.__name__

    if is_labh_datahandle_reference(the_object):
        kwargs=update_dict(kwargs, the_object.config, overwrite_if_conflict=False, interpret_none_as_val=True)
        the_object=the_object.df

    if isinstance(the_object, pd.DataFrame):
        labh_dict['filename'] = f"{var_name}.pkl"
        labh_dict['n_rows'] = len(the_object)
        labh_dict['columns'] = ', '.join(list(the_object.columns))
    
    else:
        raise NotImplementedError(f"Object of type {type(the_object)} not supported.")
    
    labh_dict=update_dict(labh_dict, kwargs, overwrite_if_conflict=False, interpret_none_as_val=True)
    return labh_dict 

def get_labh_dict_from_local(the_object:object, var_name:str, **kwargs):
    labh_dict=dict()
    labh_dict['class_parts']=get_class_parts(the_object)
    labh_dict['module_path']=getattr(the_object,'__module__', None)

    # if local is not saved, the id will be removed. So when reloading it is not tried to load a potential non-existing file/ other object
    if not getattr(the_object,'is_saved', True):
        if hasattr(the_object, 'id'):
            delattr(the_object, 'id')

    if hasattr(the_object, 'get_config'):
        kwargs=update_dict(kwargs, the_object.get_config(), overwrite_if_conflict=False, interpret_none_as_val=True)
    elif hasattr(the_object, '__dict__'):
        kwargs=update_dict(kwargs, the_object.__dict__, overwrite_if_conflict=False, interpret_none_as_val=True)

    labh_dict=update_dict(labh_dict, kwargs, overwrite_if_conflict=False, interpret_none_as_val=True)
    return labh_dict

def get_labh_dict_from_global(the_object:object, var_name:str, **kwargs):
    assert is_labh_object(the_object), f"{the_object} is not a valid labhandler object."
    the_object.update_config(kwargs)

    return the_object.get_config()

def get_labh_dict_from_object(the_object:object, var_name:str, save_file:bool=False, save_global:bool=False, **kwargs):

    if save_file:
        return get_labh_dict_from_file(the_object, var_name, **kwargs)

    if save_global:
        return get_labh_dict_from_global(the_object, var_name, **kwargs)
    
    if not (save_file or save_global):

        if isinstance(the_object, list):
            return [get_labh_dict_from_object(v, var_name, save_file, save_global, **kwargs) for v in the_object]

        return get_labh_dict_from_local(the_object, var_name, **kwargs)


###################

def load_pickle(filename:str, path:str='.', type:str=None, **kwargs) -> Any:
    load_path=(Path(path)/filename).as_posix()

    if not os.path.exists(load_path):
        logging.debug(f"{load_path} not found. Return 'None'.")
        return None

    if type=='DataFrame':
        return pd.read_pickle(load_path)
    else:
        with open(load_path, "rb") as f:
            return pickle.load(f)

def save_pickle(value:Any, filename:str, path:str='.', type:str=None, overwrite:bool=True, **kwargs) -> None:
    save_path=(Path(path)/filename).as_posix()

    if (os.path.exists(save_path)) and (not overwrite): return #do not overwrite existing files

    if type=='DataFrame':
        value.to_pickle(save_path)
    
    else:
        with open(save_path, "wb") as f:
            pickle.dump(value, f)
    
    return

def get_config_from_file(key:str, value:object, **kwargs) -> dict:
    config=dict(
        filename=f"{key}.pkl",
        type=value.__class__.__name__,
    )
 
    if config['type']=='DataFrame':
        config['columns'] = ', '.join(list(value.columns))
        config['n_rows'] = len(value)

    elif config['type']=='ndarray':
        config['shape'] = str(value.shape)
        config['dtype'] = str(value.dtype)
    
    elif config['type']=='Node':
        config['size'] = value.size
        config['height'] = value.height
    
    config=update_dict(config, kwargs, overwrite_if_conflict=False, interpret_none_as_val=True)
    return config

def get_config_from_local(key: str, value:object, **kwargs) -> dict:
    config=dict()
    config['class_parts']=kwargs.get('class_parts', get_class_parts(value))
    config['module_path']=kwargs.get('module_path',getattr(value,'__module__', None))

    # if local is not saved, the id will be removed. So when reloading it is not tried to load a potential non-existing file/ other object
    if not getattr(value,'is_saved', True):
        if hasattr(value, 'id'):
            delattr(value, 'id')

    if hasattr(value, 'get_config'):
        kwargs=update_dict(kwargs, value.get_config(), overwrite_if_conflict=False, interpret_none_as_val=True)
    elif hasattr(value, '__dict__'):
        kwargs=update_dict(kwargs, value.__dict__, overwrite_if_conflict=False, interpret_none_as_val=True)

    config=update_dict(config, kwargs, overwrite_if_conflict=False, interpret_none_as_val=True)
    return config

def get_config_from_global(key: str, value:object, **kwargs) -> dict:
    assert is_labh_object(value), f"{value} is not a valid labhandler object."
    value.update_config(kwargs)
    return value.get_config()

def get_config_from_handle(save_file:bool=False, save_global:bool=False, **kwargs) -> dict:

    if save_file:
        return get_config_from_file(**kwargs)

    if save_global:
        return get_labh_dict_from_global(**kwargs)
    
    if not (save_file or save_global):

        if isinstance(kwargs['value'], list):
            values=kwargs.pop('value')
            return [get_config_from_handle(value=v, **kwargs) for v in values]

        return get_config_from_local(**kwargs)

###

def get_class_parts(the_class):
    if not isinstance(the_class, type): the_class=the_class.__class__
    class_parts=[c.__name__ for c in the_class.mro()[:-1]][::-1]
    return class_parts

def get_class(class_name, module_path):
    """Ensure the class is loaded only once from sys.modules"""
    
    if module_path in sys.modules:
        module = sys.modules[module_path]  # ✅ Get the already loaded module
    else:
        module = importlib.import_module(module_path)  # Import only if not in sys.modules

    return getattr(module, class_name)


class Labhandler(object):
    logger = logging.getLogger(__name__)

    def load_from_lab(self, path:str=None) -> None:
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

        if not is_labh_dict(the_dict):
            return 

        self.id=the_dict.pop('id')
        self.class_parts=the_dict.pop('class_parts')
        self.module_path=the_dict.pop('module_path')
        self.update_config(the_dict)

    def load_from_path(self, path: Union[str, Path]) -> None:
        the_dict = get_yaml(Path(path)/'config.yaml')
        self.load_from_dict(the_dict)
    
    def load_from_id(self, id=None) -> None:

        if isinstance(id, int) and id<0:
            self.id=get_max_id(self._class_path)+id+1
        
        elif isinstance(id, int):
            self.id=id

            if not self.is_saved:
                warnings.warn(f"ID {self.id} not found in {self._class_path}")
                #raise FileNotFoundError(f"ID {self.id} not found in {self._class_path}")
        else:
            self.id=get_max_id(self._class_path)+1
        
        return

    def load(self, ref=None, obj=None, label:str=None) -> None:

        if obj is not None: self.load_from_object(obj)

        if is_labh_dict(ref): self.load_from_dict(ref)
        elif isinstance(ref, (str, Path)): self.load_from_path(ref)
        elif isinstance(ref, (int, type(None))): self.load_from_id(id=ref)


        if self.is_saved: #update label if it was renamed
            fn=get_filename_from_id(self._class_path, self.id, warn=True)
            label=label or re.match(ID_PATTERN, fn).group(2)
        
        self.logger.debug(f"{label=}")
        self.logger.debug(f"{getattr(self, 'label', None)=}")

        self.label = label or getattr(self, 'label', None) or self.class_parts[-1]
        self._name=f"{self.id:04d}_{self.label}"

        if self.is_saved:
            self.logger.debug(f"Loaded {self._name} from {self._path}")
            config_dict = get_yaml(f"{self._path}/config.yaml")
            config_dict = update_dict(config_dict, self.config, overwrite_if_conflict=True, interpret_none_as_val=True)
            self.update_config(config_dict)
            self.save_config()
        return

    def init_from_path(self, path:str=None, **kwargs):
        self.load(path, **kwargs)
        return

    def init_from_parent(self, locals: dict, **kwargs):
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

        self.logger.debug(f"{locals.get('label', None)=}")
        self.logger.debug(f"{locals_kwargs.get('label', None)=}")
        self.logger.debug(f"{kwargs.get('label', None)=}")
        self.logger.debug(f"{getattr(self,'label', None)=}")

        #search for label in locals, kwargs, and kwargs['kwargs']
        label=(locals.pop('label', None)) or (locals_kwargs.get('label', None)) or (kwargs.get('label', None))

        self._handled_objects=[] #reset handled objects, as they cause mutable default arguments over various reinitializations
        self.logger.debug(f"resetted {self._handled_objects=} ")

        self.load(ref, obj, label)

        #Attach Locals to target object
        for k,v in locals.items():
            setattr(obj, k, v)

        #Attach Lab Attributes to target object
        for k,v in vars(self).items():
            if (k in locals): continue
            setattr(obj, k, v)

        setattr(obj, 'logger', self.logger)

        for method_name in self._attach_methods_to_parent:
            if hasattr(obj, method_name): continue #method already exist --> do not overwrite it!
            method = getattr(self.__class__, method_name, None)

            if isinstance(method, property):
                setattr(obj.__class__, method_name, method)

            if callable(method):
                setattr(obj, method_name, MethodType(method, obj))

        return

    def __init__(self, ref=None, **kwargs):
        self.logger.debug(f"{self.__dict__=}")
        self.load_from_lab(kwargs.get('lab_path', None))
        self.logger.debug(f"{ref=}")

        self._datetime_init=get_datetime()
        self._config_key_order = ['id','label', 'lab_name','class_parts','module_path']

        self._handled_parameters=[]
        self._handled_attributes=[]

        self._handled_objects=[] #legacy

        self._attach_methods_to_parent = [
            'get_config', 'get_filtered_config', 'update_config', #config methods
            'save', 'save_config','save_files','save_attribute', #save methods
            'get_overview', '__repr__', #utility methods
            'config','path','class_path','_path','_class_path','is_saved','df' #property methods
            ]

        self._config_exclude_keys = ['labh','logger','df','model','data']+self._attach_methods_to_parent

        if isinstance(ref, str):
            if Path(ref).exists():
                self.init_from_path(ref, **kwargs)
        
        elif isinstance(ref, dict): #init from parents locals
            if all([key in ref for key in ['self']]):
                self.logger.debug(f"init from locals")
                self.handle_parent(ref)
                #self.init_from_parent(ref, **kwargs)
          
        # elif isinstance(ref, str):
        #     if Path(ref).exists():
        #         self.init_from_path(ref, **kwargs)
        
        return

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

                if keypath=='_path':
                    value=labh._path

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

    def get_filtered_config(self, the_config: dict) -> dict:
        the_config=filter_dict_keylist(the_config, self._config_exclude_keys, invert=True)
        the_config=filter_dict_keypatterns(the_config, [r'^_'], invert=True)

        handled_keys=[o['var_name'] for o in self._handled_objects]
        the_config=filter_dict_keylist(the_config, handled_keys, invert=True)

        the_config=filter_dict_valuetypes(the_config,valuetypes=[str,int,float,bool,dict,list,tuple,set,type(None)])
        the_config=filter_dict_values(the_config, [None], invert=True)

        return the_config

    def get_config(self):
        config_dict=dict()

        for k in self._config_key_order:
            if k in config_dict: continue
            if hasattr(self, k): config_dict[k]=getattr(self, k)

        config_dict.update(self.get_filtered_config(self.__dict__))

        for handle_dict in self._handled_parameters+self._handled_attributes:
            value=getattr(self, handle_dict['key'], None)
            handle_config=get_config_from_handle(value=value,**handle_dict)
            
            if isinstance(handle_config, list):
                handle_config=[self.get_filtered_config(o) for o in handle_config]
            elif isinstance(handle_config, dict):
                handle_config=self.get_filtered_config(handle_config)

            config_dict[handle_dict['key']]=handle_config

        for object_dict in self._handled_objects:

            object_config_dict=get_labh_dict_from_object(**object_dict)
            if isinstance(object_config_dict, list):
                object_config_dict=[self.get_filtered_config(o) for o in object_config_dict]
            elif isinstance(object_config_dict, dict):
                object_config_dict=self.get_filtered_config(object_config_dict)

            config_dict[object_dict['var_name']]=object_config_dict

        return config_dict

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

    def save_config(self):
        if not self.is_saved:
            self.logger.debug(f"{self} is not saved. Can not save config.")
            return
        set_yaml(self.config,f"{self._path}/config.yaml")
    
    def save_attribute(self, key:str, **kwargs) -> None:
        if not self.is_saved: 
            self.logger.debug(f"{self} is not saved. Can not save attribute '{key}'.")
            return

        if key not in self.config: #attribute is not handled yet
            self.handle_attribute(key, **kwargs)

        handle_config=self.config[key]

        value=getattr(self, key, None)
        if is_empty(value):
            self.logger.warning(f"Attribute '{key}' is empty. Skipping saving.")
            return

        save_pickle(value, path=self._path, **handle_config)
        return
  
    def save_files(self):
        if not self.is_saved:
            self.logger.debug(f"{self} is not saved. Can not save files.")
            return

        for key, handle_config in self.config.items():
            if is_labh_reference(handle_config, LABH_FILE_KEYS):
                value=getattr(self,key,None)

                if is_empty(value):
                    self.logger.debug(f"Attribute '{key}' is empty. Skipping.")
                    continue

                save_pickle(value=value, path=self._path, **handle_config)
    
    def save(self):
        os.makedirs(self._path, exist_ok=True)

        self.save_config()
        self.save_files()

    def handle_parent(self, locals:dict) -> None:

        if hasattr(self, '_parent'): self.logger.warning(f"A parent was already attached.")

        ref=locals.pop('ref', None)
        label=locals.pop('label', None)
        self._parent=locals.pop('self', None)
        del locals

        # self._handled_objects=[] #reset handled objects, as they cause mutable default arguments over various reinitializations
        # self.logger.debug(f"resetted {self._handled_objects=} ")

        self.load(ref=ref, obj=self._parent, label=label)

        #Attach labhandler attributes to parent
        for k,v in vars(self).items():
            if not is_empty(getattr(self._parent, k, None)):
                self.logger.debug(f"Attribute already exists in {self._parent.__class__.__name__}. {k=} {v=}\tSkipping attachment.")
                continue
            #if (k in locals): continue
            setattr(self._parent, k, v)
        
        #Attach labhandler methods to parent


        for method_name in self._attach_methods_to_parent:
            if hasattr(self._parent, method_name):
                self.logger.debug(f"Method '{method_name}' already exists in parent. Skipping attachment.")
                continue

            method = getattr(self.__class__, method_name, None)

            if isinstance(method, property):
                setattr(self._parent.__class__, method_name, method)

            if callable(method):
                setattr(self._parent, method_name, MethodType(method, self._parent))

        return
    
    def handle_parameter(self, locals:dict, key:str, save_file:bool=False, save_global:bool=False, **kwargs) -> Any:

        parameter=locals.pop(key, None); del locals

        if is_empty(parameter):
            parameter=self.config.get(key, None)
            
        if is_empty(parameter):
            self.logger.debug(f"'{key}' of {self._parent} is empty.")
            return None #dont handle empty parameters
        
        if isinstance(parameter, list):
            pop_len=len(self._handled_parameters)
            parameter=[self.handle_parameter({key: v}, key, save_file, save_global, **kwargs) for v in parameter]
            while len(self._handled_parameters)>pop_len: self._handled_parameters.pop() #pop back to initial lenght to avoid duplicates
        
        else:
            if is_labh_reference(parameter, LABH_LOCAL_KEYS):
                parameter=init_from_labh_reference(parameter, **kwargs)

                if is_labh_reference(parameter, LABH_GLOBAL_KEYS):
                    if parameter.is_saved and save_global:
                        warnings.warn(f"{key} is already saved globally. Setting save_global to False.")
                        save_global=False

                if is_labh_datahandle_reference(parameter):
                    kwargs.update(parameter.config)
                    parameter=parameter.df
            
            elif is_labh_dict(parameter, LABH_FILE_KEYS):
                kwargs.update(parameter)
                parameter=load_labh_file_from_dict(parameter, self._path)

        self._handled_parameters+=[dict(key=key, save_file=save_file, save_global=save_global, **kwargs)]
        return parameter
    
    def handle_attribute(self, key:str, **kwargs) -> None:

        if key in self.config:
            handle_config=copy.deepcopy(self.config[key])

            if is_labh_reference(handle_config, LABH_FILE_KEYS):
                handle_config.update(kwargs)
                value=load_pickle(path=self._path, **handle_config)

                if not is_empty(value):
                    setattr(self._parent, key, value)
                    self.logger.debug(f"Loaded attribute '{key}' from {self._path} and set to {self._parent=}")

        self._handled_attributes+=[dict(key=key, save_file=True, save_global=False, **kwargs)]
        return 

    def handle_attributes(self, keys:List[str], **kwargs) -> None:
        for key in keys: self.handle_attribute(key)
        return

    def handle_object(self, locals:dict, var_name:str, save_file:bool=False, save_global:bool=False, **kwargs):
        """
        Returns initialized object and saves it if necessary. Should be called after 'attach_parent'!.

        Args:
            locals: Dictionary containing the local variables 'locals()' of parents '__init__()'.
            var_name: Name of the variable to be initialized.
            save_file: If True, the object is saved locally in the directory of the parent
            save_global: If True, the object is saved globally as new labhandler instance.. not implemented yet.
            kwargs: Additional keyword arguments.
        """

        the_object=locals.pop(var_name, None)
        local_kwargs=locals.pop('kwargs', {})

        if is_empty(the_object):
            the_object=self.config.get(var_name, None)
        if is_empty(the_object):
            parent_label=getattr(locals.get('self', None), 'label', None)
            self.logger.debug(f"'{var_name}' of {parent_label} is empty.")
            return None #dont handle empty objects

        if save_global and save_file: warnings.warn(f"Both save_global and save_file are set to True!")

        return_object, handle_object=the_object, the_object
        if isinstance(the_object, list):
            pop_len=len(self._handled_objects)
            return_object=[self.handle_object({var_name: v}, var_name, save_file, save_global,**local_kwargs, **kwargs) for v in the_object]
            handle_object=return_object
            while len(self._handled_objects)>pop_len: self._handled_objects.pop() #pop back to initial lenght to avoid duplicates

        elif is_labh_dict(the_object, LABH_FILE_KEYS):
            return_object=load_labh_file_from_dict(the_object, self._path)
            handle_object=return_object
        
        elif is_labh_reference(the_object, LABH_LOCAL_KEYS):
            return_object=init_from_labh_reference(the_object, **local_kwargs, **kwargs)
            handle_object=return_object

            if is_labh_reference(return_object, LABH_GLOBAL_KEYS):
                if return_object.is_saved and save_global:
                    warnings.warn(f"{var_name} is already saved globally. Setting save_global to False.")
                    save_global=False

            if is_labh_datahandle_reference(return_object):
                handle_object=copy.deepcopy(return_object)
                return_object=return_object.df

        
        else:
            if save_global:
                warnings.warn(f"{var_name} ({type(the_object)=}) can not be saved globally. Setting save_global to False.")
                save_global=False
        
        self._handled_objects+=[dict(var_name=var_name, the_object=handle_object, save_file=save_file, save_global=save_global, **kwargs)]
        return return_object
        

    @property
    def config(self):
        return self.get_config()

    @property
    def class_path(self):
        return (Path(self.lab_name)/Path(*self.class_parts)).as_posix()

    @property
    def _class_path(self):
        return (Path(self._root_path)/Path(self.class_path)).as_posix()

    @property
    def path(self):
        fn = get_filename_from_id(self._class_path, self.id, warn=False)
        fn = fn or self._name
        
        return (Path(self.class_path)/fn).as_posix()

    @property
    def _path(self):
        return (Path(self._root_path)/Path(self.path)).as_posix()

    @property
    def is_saved(self):
        fn=get_filename_from_id(self._class_path, self.id, warn=False)
        return bool(fn)

    @property
    def df(self) -> pd.DataFrame:

        for object_dict in self._handled_objects:
            var_name=object_dict['var_name']
            the_object=object_dict['the_object']

            if isinstance(the_object, pd.DataFrame):
                return getattr(self, var_name, pd.DataFrame()).copy()

        #warnings.warn(f"No DataFrame Object found in {self}")
        return
                
    def __repr__(self):

        if hasattr(self, 'class_parts') and hasattr(self, 'path'):
            return f"{self.class_parts[-1]} AT: {self.path}"
        else:
            return super().__repr__()


            
        














