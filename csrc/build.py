

from pathlib import Path

import soundfile as sf
import numpy as np

from csrc.configurations import DatasetConfig
from utils import mono_load, vb

def _tagging(clip_ending_second, onsets, offsets):
   
    assert int(clip_ending_second)%DatasetConfig.dataset_clip_time == 0., 'Sorry, there is a length mismatch when trying to tagging the clip.'
    
    tag = 0
    clip_beginning_second = clip_ending_second - 5
    for _, dialogue in enumerate(zip(onsets, offsets)):
        
        if dialogue[0] > clip_ending_second: break
        
        if clip_beginning_second < dialogue[0] and clip_ending_second > dialogue[0]:
            tag = 1
            break
        
        if clip_beginning_second > dialogue[1] and clip_ending_second < dialogue[1]:
            tag = 1 
            break
        
    return tag

def _resample(y, name, clip_format, index, path, verbose=False):
    output_path = f'{path}/{index}-{name}.{clip_format}'
    if verbose:
        vb("Making file:", output_path, verbose)
    sf.write(output_path, y, DatasetConfig.dataset_sample_rate, format=clip_format, subtype='PCM_24')

def generate_from_audio(audio_path, sub_path, dest_path, sub_decoder, verbose=False):
    
        
   
    video_name = audio_path.stem if isinstance(audio_path, Path) else Path(audio_path).stem
    
  
    print("Extracting timestamps from the subtitle file...")
    decoder = sub_decoder(sub_path, encoding=DatasetConfig.sub_encoding)
    onsets, offsets = decoder.time_series
    if verbose:
        vb('Onset timestamps generated:', onsets, verbose)
        vb('Offset timestamps generated:', offsets, verbose)
        
    print("Extraction complete!\n")

    print("Librosa loading audio...")
    y, sr = mono_load(audio_path)
    print(f"Loading source file success! Using sampling rate {DatasetConfig.dataset_sample_rate}.\n")
    
   
    clip_sample_length = sr * DatasetConfig.dataset_clip_time
    
    
    clip_flag = 0 
    idx = 0 
    exceeded = False 
    sound = y
    tag_1 = 0
    tag_0 = 0

    print("Start building process.\nTransforming dataset...\n")
    while True:
        if clip_flag + clip_sample_length > len(sound):
            padding = clip_flag + clip_sample_length - len(sound)
            clip = np.zeros(clip_sample_length)
            clip[:-padding] = sound[clip_flag:]
            exceeded = True
        else:
            clip = sound[clip_flag: clip_flag+clip_sample_length]
            
        clip_flag += clip_sample_length
        idx +=1
         
       
        clip_tag = _tagging(clip_flag/sr, onsets, offsets)
        if clip_tag == 0:
            tag_0 += 1
        if clip_tag == 1:
            tag_1 += 1
            
        clip_name = video_name + '-' + str(clip_tag)
        _resample(clip, clip_name, DatasetConfig.dataset_audio_format, idx, dest_path, verbose)
        
      
        if exceeded: break
    
    print(f"Building process finished for {video_name}.")
    print(f"Label 1 (speech) clips: {tag_1}\nLabel 0 (non-speech) clips: {tag_0} \n")
    