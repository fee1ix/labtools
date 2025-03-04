
import logging
import pandas as pd
from datasets import Dataset
from typing import Union


from labtools.labhandler import Labhandler
from labtools.directory_utils import *

class Datahandle(object):

    def __init__(
            self,
            ref=None,
            label:str=None,
            remarks:str=None,
            df:pd.DataFrame=None,
            labh=Labhandler(),
            **kwargs):

        self.remarks=remarks

        if labh is not None:
            self.labh=labh
            self.labh.attach_parent(locals())

        
        if isinstance(df, pd.DataFrame):
            self.df = df.copy(); del df

        elif Path(f"{self._path}/df.pkl").exists():
            self.df = pd.read_pickle(f"{self._path}/df.pkl")
        
        if hasattr(self, 'df'):
            self.n_rows = len(self.df)
            self.columns = self.df.columns
        
        self.n_rows = len(self.df)
        self.columns = list(self.df.columns)

    
    def spawn(self):
        self.spawn_config()
        self.df.to_pickle(f"{self._path}/df.pkl")
        


            


