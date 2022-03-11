
import os



import ffmpeg


ROOT_PATH_ABS = os.path.split(os.path.realpath(__file__))[0]
OUTPUT_FOLDER_ABS = ROOT_PATH_ABS + "/output"
RESULT_FOLDER_ABS = ROOT_PATH_ABS + "/results"
TEMP_FOLDER_ABS = ROOT_PATH_ABS + "/temp"

INFERENCE_PARAMS_PATH = ROOT_PATH_ABS + "/models/model_best.pth"


class DecoderConfig(object):
   
    
    trimming_end = 0
    trimming_start = 0


class PostProcessConfig(object):
    
    standard_dialogue_break = 0
    
    loose_dialogue_threshold = 2 * standard_dialogue_break
    loose_dialogue_delay = loose_dialogue_threshold / 4
    
  
    global_bias = 0.25

   
    max_sigle_speech_length = 1
    
    
class InferenceConfig(object):
    
    best_around_period = 1
    
   
    threshold = 0.3
    
  
    coding_map = {
        0: "non-speech",
        1: "speech",
    }



class SSourceConfig(object):
   
    
    headers = {
        "Title": None,
        "Original Script": "ASFG",
        "PlayResX": None,
        "PlayResY": None,
        "Timer": 100.0000,
    }
    
    v4plus_pairs = {
        "Name": "chs",
        "Fontname": "simhei",
        "Fontsize": 20,
        "PrimaryColour": "&H00ffffff",
        "SecondaryColour": "&H0000ffff",
        "OutlineColour": "&H00000000",
        "BackColour": "&H80000000",
        "Bold": 1,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 90,
        "ScaleY": 90,
        "Spacing": 0,
        "Angle": 0.00,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 2,
        "Alignment": 2,
        "MarginL": 20,
        "MarginR": 20,
        "MarginV": 15,
        "Encoding": 1,
    }
    
    v4_pairs = {
        "Name": "eng",
        "Fontname": "Arial Narrow",
        "Fontsize": 12,
        "PrimaryColour":"&H00ffeedd",
        "SecondaryColour": "&H00ffc286",
        "TertiaryColour": "&H00000000",
        "BackColour": "&H80000000",
        "Bold": -1,
        "Italic": 0,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 0,
        "Alignment": 2,
        "MarginL": 20,
        "MarginR": 20,
        "MarginV": 2,
        "AlphaLevel": 0,
        "Encoding": 1,
    }
    
   
    events_pairs = {
        "Marked": 0,
        "Start": None,
        "End": None,
        "Style": None,
        "Name": "",
        "MarginL": "0000",
        "MarginR": "0000",
        "MarginV": "0000",
        "Effect": "",
        "Text": "xxx",
    }
    
