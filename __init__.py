import os
import logging
import warnings

from types import MethodType
from functools import partial
from typing import Optional, Union

from labtools.directory_utils import *
from labtools.dictionary_utils import *

LAB_SUFFIX='_lab'


def get_cls_parts(cls):
    if not isinstance(cls, type): cls=cls.__class__
    cls_parts=[c.__name__ for c in cls.mro()[:-1]][::-1]
    return cls_parts


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
        self.cls_parts=get_cls_parts(obj)

        cls_path=(Path(self.lab_name)/Path(*self.cls_parts)).as_posix()
        _cls_path=(Path(self._root_path)/Path(cls_path)).as_posix()

        os.makedirs(_cls_path, exist_ok=True)

        if isinstance(ref, int):
            self.id=ref
        elif ref is None:
            self.id=get_max_id(_cls_path)+1

        self.name=f"{self.id:04d}_{self.cls_parts[-1]}"
        self.path=(Path(cls_path)/Path(self.name)).as_posix()
        self._path=(Path(_cls_path)/Path(self.name)).as_posix()

        if Path(self._path).exists():
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

        self._config_key_order=['name', 'lab_name', 'cls_parts','id','path']

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

    def attach(self, locals, **kwargs):
        del locals['lab_handler']
        obj=locals.pop('self', None)
        ref=locals.pop('ref', None)
        
        self.load_ref(ref, obj)

        #Attach Lab Attributes to target object
        for k,v in vars(self).items(): setattr(obj, k, v)

        #Attach Lab Methods to target object
        methods_to_attach = ['get_config', 'spawn', 'spawn_config','update_config']

        for method_name in methods_to_attach:
            method = getattr(self.__class__, method_name, None)
            if callable(method):
                setattr(obj, method_name, MethodType(method, obj))
            
        














