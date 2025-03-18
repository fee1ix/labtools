import re
import copy
import logging
from typing import List, Any

def split_keypath(keypath):
    if isinstance(keypath,str):
        keypath_list=re.split(r'[\./]',keypath)
    elif isinstance(keypath,list):
        keypath_list=keypath
    return keypath_list

def is_nested_dict(a_dict):
    return any(isinstance(v, dict) for v in a_dict.values())

def is_unique(element, a_list):
    return a_list.count(element) == 1

def has_duplicates(a_list):
    return len(a_list) != len(set(a_list))

def get_duplicates(a_list):
    return list(set([x for x in a_list if a_list.count(x) > 1]))

def get_keys(the_dict, atomic_only=False):
    """
    Get all keys from a dictionary. If atomic_only is True, only return keys that are not associated with a nested dictionary.
    """
    keys = []
    for k, v in the_dict.items():
        if not atomic_only or not isinstance(v, dict):
            keys.append(k)
        if isinstance(v, dict):
            keys.extend(get_keys(v, atomic_only))
    return keys

def get_keypaths(the_dict:dict, atomic_only:bool=False) -> List[List[str]]:
    """
    Get all keys from a dictionary. If atomic_only is True, only return keys that are not associated with a nested dictionary.
    """
    assert isinstance(the_dict, dict), "Input must be a dictionary."

    def _recursive(current_dict: dict, current_path: List[str] = []) -> List[List[str]]:
        keypaths = []
        for key, value in current_dict.items():
            new_path = current_path + [key]
            if isinstance(value, dict):
                # Continue recursion for nested dictionaries
                keypaths.extend(_recursive(value, new_path))
                if not atomic_only:
                    keypaths.append(new_path)
            else:
                # Only add atomic paths if atomic_only is True
                keypaths.append(new_path)
        return keypaths

    return _recursive(the_dict)

def get_attrpaths(the_object: object, atomic_only: bool = False, max_depth:int=5) -> List[List[str]]:
    """
    Get all attribute paths from an object. If atomic_only is True, only return paths that end in non-object (atomic) values.
    """
    raise NotImplementedError("This function is not yet implemented.")
    assert isinstance(the_object, object), "Input must be an object."

    def _recursive(current_object: Any, current_path: List[str] = []) -> List[List[str]]:
        paths = []

        if len(current_path) > max_depth:
            return paths

        for attr in dir(current_object):
            if attr.startswith('_') or callable(getattr(current_object, attr)):
                continue
            value = getattr(current_object, attr)
            new_path = current_path + [attr]
            if hasattr(value, '__dict__'):
                paths.extend(_recursive(value, new_path))
                if not atomic_only:
                    paths.append(new_path)
            else:
                paths.append(new_path)
        return paths

    return _recursive(the_object)
    



    # keys = []
    # _keypaths=[]
    # for k, v in the_dict.items():
    #     if not atomic_only or not isinstance(v, dict):
    #         keys.append(k)


    #     if isinstance(v, dict):
    #         keys.extend(get_keys(v, atomic_only))
    # return keys


def get_attrs(the_object:object, atomic_only:bool=False) -> list:
    """
    Get all attributes from an object. If atomic_only is True, only return attributes that are not associated with a nested object.
    """
    assert isinstance(the_object, object), "Input must be an object."

    attrs=[]

    return attrs


def has_keys(the_dict, keys, atomic_only=False):
    return all([k in get_keys(the_dict, atomic_only=atomic_only) for k in keys])

def has_key(the_dict, key, atomic_only=False):
    return key in get_keys(the_dict, atomic_only=atomic_only)


def set_value(obj:object, attr_or_key: str, value: Any) -> None:

    if isinstance(obj, dict):
        obj[attr_or_key] = value
    else:
        setattr(obj,attr_or_key,value)
    
    return

def hasattr_or_key(obj:object, attr_or_key:str) -> bool:
    # Check if it's a dictionary and contains the key
    if isinstance(obj, dict):
        return attr_or_key in obj
    # Otherwise, check if it's an object and has the attribute
    return hasattr(obj, attr_or_key)

def hasattrs_or_keys(obj:object, attrs_or_keys: list) -> bool:
    return all(hasattr_or_key(obj, key) for key in attrs_or_keys)

def get_value(obj: object, attr_or_key:str, value:Any=None) -> Any:
    # Check if it's a dictionary and contains the key
    if isinstance(obj, dict):
        if attr_or_key in obj:
            value = obj[attr_or_key]
    else:
        value = getattr(obj, attr_or_key, value)

    return value

def invoke_dict_callable(the_dict: dict, callable_name: str) -> dict:

    def _recursive(data, callable_name):
        if isinstance(data, dict):  # If it's a dictionary, process each key-value pair
            return {key: _recursive(value, callable_name) for key, value in data.items()}
        
        elif isinstance(data, list):  # If it's a list, process each element
            return [_recursive(item, callable_name) for item in data]

        elif hasattr(data, callable_name):  
            attr = getattr(data, callable_name)
            if callable(attr):  # If it's a method, call it
                return attr()
            else:  # If it's an attribute (like __dict__), return it directly
                return attr
            
        return data

    return _recursive(the_dict, callable_name)  # Otherwise, return as is

def filter_dict_valuetypes(the_dict: dict, valuetypes: list = [], invert: bool = False) -> dict:
    """
    Filters a dictionary recursively based on the types of its values.
    Args:
        the_dict (dict): The input dictionary to be filtered.
        valuetypes (list, optional): A list of value types to filter by. Defaults to an empty list.
        invert (bool, optional): If True, the function will exclude values of the specified types 
                                    instead of including them. Defaults to False.
    Returns:
        dict: A new dictionary containing only the key-value pairs that match the filtering criteria.
    Notes:
        - If `invert` is False, the function includes only the key-value pairs where the value's type 
            matches one of the types in `valuetypes`.
        - If `invert` is True, the function excludes key-value pairs where the value's type matches 
            one of the types in `valuetypes`.
        - The function processes nested dictionaries recursively.
        - Empty dictionaries and lists are removed from the result.
    Example:
        >>> input_dict = {
        ...     "a": 1,
        ...     "b": "string",
        ...     "c": {"d": 2, "e": []},
        ...     "f": [],
        ... }
        >>> filter_dict_valuetypes(input_dict, valuetypes=[int])
        {'a': 1, 'c': {'d': 2}}
    """

    if (dict in valuetypes) and not invert: valuetypes.remove(dict)
    if (dict not in valuetypes) and invert: valuetypes.append(dict)

    def _recursive(the_dict):

        new_dict=dict()

        for k,v in the_dict.items():

            if isinstance(v, tuple(valuetypes)) ^ invert:
                new_dict[k]=v

            elif isinstance(v, dict) and not invert:
                    new_dict[k] = _recursive(v)
            
            val=new_dict.get(k,{})
            if isinstance(val, (dict, list)) and (len(val)==0):
                new_dict.pop(k,None)

            # elif isinstance(v, list) and list in valuetypes and not invert:
            #     # Process list elements individually
            #     new_dict[k] = [item for item in v if isinstance(item, tuple(valuetypes))]
            
        return new_dict
    
    return _recursive(the_dict)



def filter_dict_values(the_dict, values=[], invert=False):
    def _recursive(the_dict):
        new_dict = {}
        for k, v in the_dict.items():
            if isinstance(v, dict):
                # Recursively filter nested dictionaries
                filtered_subdict = _recursive(v)
                if filtered_subdict or (v in values) ^ invert:
                    new_dict[k] = filtered_subdict
            elif (v in values) ^ invert:
                # Include the value if the condition matches
                new_dict[k] = v
        return new_dict

    return _recursive(the_dict)

def filter_dict_valuepatterns(the_dict, valuepatterns=[], invert=False):

    def _recursive(the_dict):
            
        new_dict=dict()
        for k,v in the_dict.items():

            if any([re.match(valuepattern, str(v)) for valuepattern in valuepatterns]) ^ invert:

                if isinstance(v, dict):
                    new_dict[k]=_recursive(v)
                
                else:
                    new_dict[k]=v
 
        return new_dict
    
    return _recursive(the_dict)

def filter_dict_keypatterns(the_dict, keypatterns=[], invert=False):

    def _recursive(the_dict):
            
        new_dict=dict()
        for k,v in the_dict.items():

            if any([re.match(keypattern, k) for keypattern in keypatterns]) ^ invert:

                if isinstance(v, dict):
                    new_dict[k]=_recursive(v)
                
                else:
                    new_dict[k]=v
 
        return new_dict
    
    return _recursive(the_dict)

def filter_dict_keylist(the_dict:dict, keylist:list=[], invert:bool=False) -> dict:
    """
    Filters a dictionary by including or excluding keys based on a given key list.
    This function recursively traverses a dictionary and filters its keys based on 
    the provided `keylist`. If `invert` is set to `True`, the function excludes the 
    keys in `keylist` instead of including them.
    Args:
        the_dict (dict): The dictionary to filter.
        keylist (list, optional): A list of keys to include or exclude. Defaults to an empty list.
        invert (bool, optional): If `False`, only keys in `keylist` are included. 
                                 If `True`, keys in `keylist` are excluded. Defaults to `False`.
    Returns:
        dict: A new dictionary containing the filtered keys and their associated values.
    """
    def _recursive(the_dict):
        new_dict=dict()
        for k,v in the_dict.items():

            if (k in keylist) ^ invert:

                if isinstance(v, dict):
                    new_dict[k]=_recursive(v)
                
                else:
                    new_dict[k]=v

        return new_dict
    
    return _recursive(the_dict)

def filter_dict_keydict(the_dict, ref_dict, invert=False):

    def _recursive(the_dict,ref_dict):
        new_dict=dict()
        for k,v in the_dict.items():
            if k in ref_dict:
                if isinstance(v, dict):
                    new_dict[k]=_recursive(v, ref_dict[k])
                else:
                    new_dict[k]=v

        return new_dict
    
    
    if invert:
        new_dict = filter_dict_keylist(the_dict, keylist=get_keys(ref_dict,atomic_only=False), invert=True)
    
    else:
        new_dict = _recursive(the_dict, ref_dict)

    return new_dict


def update_dict(orign_dict, mixin_dict, interpret_none_as_val=True, overwrite_if_conflict=True):
    """
    Update origin_dict with values from update_dict. If overwrite is True, values from update_dict will overwrite values from origin_dict
    """

    def _recursive(orign_dict, mixin_dict):
        for k, v in mixin_dict.items():
            # Check if both original[key] and value are dictionaries
            if k in orign_dict and isinstance(orign_dict[k], dict) and isinstance(v, dict):
                # Recurse if both are dictionaries
                orign_dict[k] = _recursive(orign_dict[k], v)

            elif (k not in orign_dict) or ((orign_dict[k] is None) and (not interpret_none_as_val)):
                orign_dict[k] = v

            elif overwrite_if_conflict:
                orign_dict[k] = v

        return orign_dict

    return _recursive(orign_dict, mixin_dict)

def has_keypath(the_dict, keypath):

    def _recursive(the_dict, keypath):
        key = keypath[0]

        if not isinstance(the_dict, dict): return False

        if not key in the_dict: return False
        if len(keypath)==1: return True
        else: return _recursive(the_dict[key], keypath[1:])

    return _recursive(the_dict, keypath)

def hasattr_or_keypath(obj:object, attr_or_keypath:list) -> bool:

    def _recursive(obj:object, attr_or_keypath: List[str]) -> bool:

        attr_or_key = attr_or_keypath[0]
        if not hasattr_or_key(obj, attr_or_key): return False
        if len(attr_or_keypath)==1: return True

        else: return _recursive(get_value(obj,attr_or_key), attr_or_keypath[1:])

    return _recursive(obj, attr_or_keypath)


def get_dict_value_from_keypath(the_dict, keypath):

    def _recursive(the_dict, keypath):
        key = keypath[0]
        if key in the_dict:
            if len(keypath)>1:
                return _recursive(the_dict[key], keypath[1:])
            else:
                return the_dict[key]
        else:
            return None
        
    return _recursive(the_dict, keypath)


def get_dict_values(the_dict: dict, key: str) -> List[Any]:
    """
    Recursively find all values for a given key in a nested dictionary.
    """
    values = []

    def _recursive(current_dict):
        if isinstance(current_dict, dict):
            for k, v in current_dict.items():
                if k == key:
                    values.append(v)
                if isinstance(v, dict):
                    _recursive(v)
                elif isinstance(v, list):
                    for item in v:
                        _recursive(item)

    _recursive(the_dict)
    return values


def getattr_or_keypath(obj:object, attr_or_keypath:List[str], value:Any=None) -> Any:

    def _recursive(obj, attr_or_keypath, value=None):
        attr_or_key = attr_or_keypath[0]

        if hasattr_or_key(obj, attr_or_key):
            if len(attr_or_keypath)>1:
                return _recursive(get_value(obj,attr_or_key), attr_or_keypath[1:])
            
            else:
                return get_value(obj,attr_or_key)
        else:
            return value
        
    return _recursive(obj, attr_or_keypath, value)



def get_dict_value(the_dict, key):
    ambiguous_keypaths=get_dict_keypaths(the_dict, key)

    value=None
    if len(ambiguous_keypaths)==1:
        value=get_dict_value_from_keypath(the_dict, ambiguous_keypaths[0])

    elif len(ambiguous_keypaths)>1:

        ambiguous_values=[]
        for ambiguous_keypath in ambiguous_keypaths:
            ambiguous_values.append(get_dict_value_from_keypath(the_dict, ambiguous_keypath))
        
        if len(set(ambiguous_values))==1:
            value=ambiguous_values[0]

        else:
            logging.warning(f"Multiple values found for '{key}' in config_dict: {ambiguous_values}")

    return value

def get_dict_keypaths(the_dict, key):

    def _recursive(the_dict, key, current_path=[]):
        keypaths = []
        for k, v in the_dict.items():
            if k == key:
                keypaths.append(current_path + [k])
            if isinstance(v, dict):
                keypaths.extend(_recursive(v, key, current_path + [k]))

        return keypaths
    
    return _recursive(the_dict, key)


def get_dict_keypath(the_dict,key):

    all_keys=get_keys(the_dict, atomic_only=False)
    #print(is_unique(key,all_keys),key,all_keys)
    if not is_unique(key,all_keys): raise KeyError(f"key '{key}' is not unique in the dictionary.")

    def _recursive(the_dict, key):
        for k, v in the_dict.items():
            if k == key:
                return [k]
            elif isinstance(v, dict):
                keypath = _recursive(v, key)
                if keypath:
                    return [k] + keypath
    
    return _recursive(the_dict, key)

def set_dict_keypath(the_dict, keypath, value, inplace=False):

    def _recursive(d, kp, val):
        if len(kp) == 1:
            d[kp[0]] = val
        else:
            if kp[0] not in d:
                d[kp[0]] = {}
            _recursive(d[kp[0]], kp[1:], val)

    if inplace:
        _recursive(the_dict, keypath, value)
        return the_dict
    else:
        new_dict=copy.deepcopy(the_dict)
        _recursive(new_dict, keypath, value)
        return new_dict

def del_dict_keypath(the_dict, keypath, inplace=False):

    def _recursive(d, kp):
        if len(kp) == 1:
            del d[kp[0]]
        else:
            if kp[0] not in d:
                d[kp[0]] = {}
            _recursive(d[kp[0]], kp[1:])

    if inplace:
        _recursive(the_dict, keypath)
        return the_dict
    else:
        new_dict=copy.deepcopy(the_dict)
        _recursive(new_dict, keypath)
        return new_dict 


def nestify_dict_like(the_dict, ref_dict):
    ref_keys = get_keys(ref_dict)
    the_keys = get_keys(the_dict)

    for k in the_keys:

        if not is_unique(k,ref_keys): continue; logging.warning(f"key '{k}' is not unique in the reference dictionary.")
        if not is_unique(k,the_keys): continue; logging.warning(f"key '{k}' is not unique in the dictionary.")

        the_keypath=get_dict_keypath(the_dict, k)
        ref_keypath=get_dict_keypath(ref_dict, k)

        if the_keypath != ref_keypath:

            the_value=get_dict_value_from_keypath(the_dict, the_keypath)
            the_dict=set_dict_keypath(the_dict, ref_keypath, the_value)
            #the_dict=set_dict_keypath(the_dict, ref_keypath, the_dict[k])
            the_dict=del_dict_keypath(the_dict, the_keypath)
            #del the_dict[k]

    return the_dict

def update_dict_smart(
        orign_dict,
        mixin_dict,
        interpret_none_as_val=True,
        overwrite_if_conflict=True,
        allow_new_atomic_keys=False,
        allow_new_nested_keys=False,
        ):
    
    mixin_atomic_keys = get_keys(mixin_dict, atomic_only=True)
    mixin_nested_keys = [k for k in get_keys(mixin_dict, atomic_only=False) if k not in mixin_atomic_keys]

    allow_keylist=get_keys(orign_dict, atomic_only=False)
    logging.debug(f"allow_keylist: {allow_keylist}")
    if allow_new_atomic_keys:
        new_atomic_keys=[k for k in mixin_atomic_keys if k not in allow_keylist]
        logging.debug(f"new_atomic_keys: {new_atomic_keys}")
        allow_keylist.extend(new_atomic_keys)

    if allow_new_nested_keys:
        new_nested_keys=[k for k in mixin_nested_keys if k not in allow_keylist]
        logging.debug(f"new_nested_keys: {new_nested_keys}")
        allow_keylist.extend(new_nested_keys)


    mixin_dict=filter_dict_keylist(mixin_dict, keylist=allow_keylist)
    mixin_dict=nestify_dict_like(mixin_dict, orign_dict)

    orign_dict=update_dict(
        orign_dict, 
        mixin_dict, 
        interpret_none_as_val=interpret_none_as_val,
        overwrite_if_conflict=overwrite_if_conflict
        )
    
    return orign_dict


def pick_from_dict(taker_dict, giver_dict):

    taker_dict=update_dict_smart(
        taker_dict,
        giver_dict,
        interpret_none_as_val=True,
        overwrite_if_conflict=True,
        allow_new_atomic_keys=False,
        allow_new_nested_keys=False,
    )

    giver_dict=filter_dict_keydict(giver_dict, taker_dict, invert=True)

    return taker_dict, giver_dict
