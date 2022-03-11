

from pathlib import Path
import os
import time
import mimetypes
import os

import librosa
import moviepy.editor as mp

def mono_load(path, sr=32000, mono=True):
   
    
    start = time.time()
    
    print(f"Loading file: {path}")
    y, c = librosa.load(path, sr=sr, mono=mono)
    
    end = time.time()
    
    print(f"Loading completed. Cost {(end-start):.2f}s\n")
    
    return y, c 
    
def vb(pre, any, v):
  
    if v:
        print(f"{pre} {repr(any)}")
        
def check_type(file_path):
    
    
    assert (Path(file_path).exists()) and (Path(file_path).is_file()), "Your input file path is not valid or the file doesn't exist."
    
    is_video = False
    
    mimetypes.init()
    
    mimestart = mimetypes.guess_type(str(file_path))[0]
    
    if mimestart: 
        try:
            mimestart = mimestart.split("/")[0]
        except RuntimeError as e:
            print(e)
            print("Unrecognizable file type. Is the file format valid\n")    
        
        assert mimestart=="video" or mimestart=="audio", "Input file format unrecognizable as video or audio (using mimetypes).\n"
        
        if mimestart == "video": is_video = True 

    return is_video

def extract_audio(file_path, format: str="wav"):
   
    print(f"Extracting audio from {file_path}")
    
    mv = mp.VideoFileClip(file_path)
    assert mv!=None, "Unable to extract any information from the video clip."
    
    mv_name = str(file_path).split("/")[-1].split(".")[0]
    mv_audio_file = Path(file_path).parent / f"{mv_name}.{format}"
    
    
    try:
        mv.audio.write_audiofile(mv_audio_file)
    except AttributeError:
        print("\nNote: Moviepy failed to resolve your video path. Currently Use Path as string for moviepy to work.\n")
        mv.audio.write_audiofile(str(mv_audio_file))
     
    print(f"Extraction Successful! Writing {Path(mv_audio_file).stat().st_size} in {mv_audio_file}.")
    
    return mv_audio_file

def count_class(path):

    
    zeros = 0
    ones = 0
    total = 0
    
    for file in os.listdir(path):
        c = str(file).split(".")[0][-1]
        ones += int(c)
        total += 1
    zeros = total - ones
    
    print(f"\nLabel 1 instances: {ones}")
    print(f"Label 0 instances: {zeros}")
    print(f"Total clips: {total}\n")
    
    return zeros, ones, total

def get_duration(audio_file_path, y=None, sr=None):
  
    
    header_duration = librosa.get_duration(filename=audio_file_path)
    
    if sr:
        wavform_duration = librosa.get_duration(y=y, sr=sr)
        
        if header_duration == wavform_duration:
            print("Audio file consistency ensured.")
        else:
            print("There is inconsistency between the audio waveform and header metadata." \
                "This could be ignored.")
            
        return wavform_duration
    
    return header_duration
    