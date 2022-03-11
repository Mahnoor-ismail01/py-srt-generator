

class ModelConfig(object):
   
    sed_model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": 527,
    }
    
class DatasetConfig(object):

  
    
    dataset_clip_time = 2 # seconds
    
    dataset_sample_rate = 32000
    
    dataset_audio_format = "wav" 
    
    sub_encoding = "utf-8" # recommanded
    