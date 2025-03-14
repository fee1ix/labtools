import sys

# LOGGING CONFIG
import logging

WORKING_MODULES = ['labtools', 'beesup_llm']
LOGGING_FORMAT = '%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'


logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT)

# Automatically suppress all other loggers
for name in logging.root.manager.loggerDict:
    if not any(name.startswith(allowed) for allowed in WORKING_MODULES):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)

import importlib

def get_modules_with_prefixpath(prefixpaths=WORKING_MODULES):
    if isinstance(prefixpaths, str):
        prefixpaths = [prefixpaths]

    modules=dict()
    all_modules = sys.modules.copy()
    for prefixpath in prefixpaths:
        for module_path, module in all_modules.items():
            if module_path.startswith(prefixpath):
                modules[module_path] = module

    return modules

def reimport(prefixpaths=WORKING_MODULES):
    for module_path, module in get_modules_with_prefixpath(prefixpaths).items():
        importlib.reload(sys.modules[module_path])
        globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})

def set_info(prefixpaths=WORKING_MODULES):
    for module_path in get_modules_with_prefixpath(prefixpaths).keys():
        logger = logging.getLogger(module_path)
        logger.setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)

def set_debug(prefixpaths=WORKING_MODULES):
    for module_path in get_modules_with_prefixpath(prefixpaths).keys():
        logger = logging.getLogger(module_path)
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
