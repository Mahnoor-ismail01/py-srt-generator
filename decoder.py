

from pathlib import Path
from config import DecoderConfig as DC


class deSRT(object):
   
    
    def __init__(self, file_path, encoding, trim) -> None:
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.trim = trim
        
    def _trim_events(self, onsets, offsets):
       
        
        def handle_offset(offset):
            if (offset % 2.) < DC.trimming_end:
                offset = float(int(offset))
            
            return offset
                
        def handle_onset(onset):
            if on_end := ((onset / 2. + 1) * 2 - onset) < DC.trimming_start:
                onset = float(on_end)
            
            return onset
            
     
        offsets = list(map(handle_offset, offsets))
        
        # Trim onsets.
        onsets = list(map(handle_onset, onsets))
        
        return onsets, offsets


             
            
class SRTD(deSRT):
    
    
    file_type = "srt"
    
    def __init__(self, file_path, encoding="utf-8", trim=True) -> None:
        assert isinstance(file_path, str) or isinstance(file_path, Path), "Invalid file path, only 'str' and Pathlib.Path' supported."
        super().__init__(file_path, encoding, trim)

    def _decode_time(self, str_time):
            
        tail = float(str_time.split(",")[-1]) * 1e-3
        h, m, s = str_time.split(",")[0].split(":")
        float_time = int(h)*3600 + int(m)*60 + int(s) + tail
        
        return float_time
        
    @property
    def time_series(self):
      
        on_ts = []
        off_ts = []
        
        with open(self.file_path, mode="r", encoding=self.encoding) as f:
            for line in f.readlines():
                if "-->" in line:
                    onset = line.split("-")[0].lstrip().rstrip()
                    offset = line.split(">")[-1].lstrip().rstrip()
                    onset = self._decode_time(onset)
                    offset = self._decode_time(offset)
                    if onset:
                        on_ts.append(onset)
                    if offset: 
                        off_ts.append(offset)
                        
        on_ts, off_ts = self._trim_events(on_ts, off_ts) if self.trim else (on_ts, off_ts)
        
        assert len(on_ts)==len(off_ts), "Mismatch for timestamp series."

        return on_ts, off_ts
