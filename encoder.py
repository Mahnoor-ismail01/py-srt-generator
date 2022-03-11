

import pandas as pd
from pandas.io.formats.format import CategoricalFormatter

from config import ROOT_PATH_ABS, SSourceConfig as SSC
from config import RESULT_FOLDER_ABS


class Encoder(object):
   
    
    def __init__(self, df:pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.start_series = [f"{self._format_time(float(fl))}" for fl in self.df.start]
        self.end_series = [f"{self._format_time(float(fl))}" for fl in self.df.end]
       
        try:
            self.texts = self.df.recognized_text
        except:
            self.texts = ['xxx'] * len(self.start_series)

    def _format_time(self, fl):
        
        int_str_part, decimal_str_part = str(fl).split(".")
        int_part = int(int_str_part)
        decimal_str_part = decimal_str_part[:2]
        
        s = int_part % 60 # seconds
        m = (int_part // 60) % 60 # minutes
        h = int_part // 3600 # hours

        return f"{h}:{m}:{s}.{decimal_str_part}"




class SRTEncoder(Encoder):
   
    
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        
    def _format_time_presentation(self, str_time):
        
        
        i, f = str_time.split(".")
        
        h, m, s = i.split(":")
        
        h = ("0" + h) if len(h)<2 else h
        m = ("0" + m) if len(m)<2 else m
        s = ("0" + s) if len(s)<2 else s

        while len(f) < 3:
            f = f + "0"

        formatted_str_time = f"{h}:{m}:{s},{f}"
        
        return formatted_str_time

    @property
    def event_timestamps(self) -> list:
        event_collections = []
        
        for (s, e) in zip(self.start_series, self.end_series):
            event_line = f"{self._format_time_presentation(s)} --> {self._format_time_presentation(e)}"
            event_collections.append(event_line)
        
        return event_collections
        
    def generate(self, file_name, target_dir=ROOT_PATH_ABS, encoding="utf-8"):
        
        
        path = f"{target_dir}/{file_name}"       
        if not "srt" in file_name:
            path = path + ".srt"
            
        with open(path, mode="w", encoding=encoding) as f:
            for (idx, (timeline, text)) in enumerate(zip(self.event_timestamps, self.texts)):
                f.write(str(idx+1))
                f.write("\n")
                f.write(timeline)
                f.write("\n")
                f.write(str(text))
                f.write("\n")

                f.write("\n")
                