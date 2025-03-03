
import logging
import pandas as pd
from datasets import Dataset

from labtools import *

class DataHandle(object):

    def __init__(self, ref=None, df:pd.DataFrame=None, remarks:str=None, lab_handler=LabHandler(), **kwargs):

        self.remarks=remarks

        if lab_handler is not None:
            self.lab=lab_handler
            self.lab.attach(locals())

        
        if isinstance(df, pd.DataFrame):
            self.df = df

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
        


            


