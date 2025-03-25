
import logging
import pandas as pd
from datasets import Dataset
from typing import Union


from labtools.labhandler import Labhandler
from labtools.directory_utils import *


class Datahandle(object):
    logger=logging.getLogger(__name__)

    def __init__(
            self,
            ref=None,
            label:str=None,
            remarks:str=None,
            data_df:pd.DataFrame=None,
            labh=Labhandler,
            **kwargs):
        
        self.logger.debug(f"{self.__dict__=}")
        
        self.label = label
        self.remarks = remarks

        if labh is not None:
            self.labh=labh(locals())
            data_df=self.labh.handle_parameter(locals(),'data_df', save_file=True, overwrite=False)
        
        if isinstance(data_df, pd.DataFrame):
            self.data_df = data_df.copy(); del data_df

class BaseDatahandle(Datahandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EmbeddingDatahandle(Datahandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ChunksDatahandle(Datahandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class InjectionDatahandle(Datahandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EvaluationDatahandle(Datahandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



        


            


