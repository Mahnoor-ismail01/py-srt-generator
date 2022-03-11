import numpy as np
from numpy.lib.utils import source
import pandas as pd
import torch
import librosa
import soundfile
from icecream import ic
from utils import check_type, extract_audio, mono_load
from config import InferenceConfig  as IC
from csrc.configurations import DatasetConfig as DC
from csrc.configurations import ModelConfig as MC
from csrc.dataset import PANNsDataset
from csrc.models import PANNsCNN14Att, AttBlock
from post import SpeechSeries
from stt.vosk_api import get_by_ffmpeg
from config import TEMP_FOLDER_ABS



PERIOD = IC.best_around_period
THRESHOLD = IC.threshold
CODING = IC.coding_map 
SR = DC.dataset_sample_rate


class Pannscnn14attInferer():
    
    def __init__(self, clip_y, model_path, period=0, device=None, ds=None):
        self.clip_y = clip_y
        self.model_path = model_path
        
        self.period = period
        self.device = device
        self.model = PANNsCNN14Att(**MC.sed_model_config)
        self.model.att_block = AttBlock(2048, 2, activation='sigmoid')
        self.model.att_block.init_weights()
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu"))['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def make_inference_result(self):
        audios, start, end = self.new_method1()

        while True:
            y_batch = self.new_method2(start, end)
            if len(y_batch) != PERIOD * SR:
                y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
                y_pad[:len(y_batch)] = y_batch
                audios.append(y_pad)
                break
            start = end
            end += PERIOD * SR
            audios.append(y_batch)

        
        arrays = np.asarray(audios)
        multiarray = torch.from_numpy(arrays)

        estimated_event_list = []
        global_time = 0.0
        for rowss in multiarray:
            rowss = rowss.view(1, rowss.size(0))
             

            rowss = rowss.to(self.device)
            
          

            with torch.no_grad():
                prediction = self.model(rowss)
                 
              
                framewise_outputs = prediction["framewise_output"].detach(
                    ).cpu().numpy()[0]
                
                    
            thshold= framewise_outputs >= THRESHOLD
            

            

            for target_idx in range(thshold.shape[1]):
                if thshold[:, target_idx].mean() == 0:
                    pass
                else:
                    detected = np.argwhere(thshold[:, target_idx]).reshape(-1)
                    
                    head_idx = 0
                    tail_idx = 0
                    while True:
                        if (tail_idx + 1 == len(detected)) or (
                                detected[tail_idx + 1] - 
                                detected[tail_idx] != 1):
                            onset = 0.01 * detected[
                                head_idx] + global_time
                            offset = 0.01 * detected[
                                tail_idx] + global_time
                            onset_idx = detected[head_idx]
                            offset_idx = detected[tail_idx]
                            max_confidence = framewise_outputs[
                                onset_idx:offset_idx, target_idx].max()
                            
                            mean_confidence = framewise_outputs[
                                onset_idx:offset_idx, target_idx].mean()
                            
                            estimated_event = {
                                "speech_recognition": CODING[target_idx],
                                "start": onset,
                                "end": offset,
                                "max_confidence": max_confidence,
                                "mean_confidence": mean_confidence,
                            }
                            estimated_event_list.append(estimated_event)
                            head_idx = tail_idx + 1
                            tail_idx = tail_idx + 1
                            if head_idx >= len(detected):
                                break
                        else:
                            tail_idx += 1
            global_time += PERIOD
        
        ic(estimated_event_list)   
        return estimated_event_list

    def new_method2(self, start, end):
        y_batch = self.clip_y[start:end].astype(np.float32)
        return y_batch

    def new_method1(self):
        audios = []
        start = 0
        end = PERIOD * SR
        return audios,start,end
    
    def get_breakpoint(self):
        pass
    

class SttInferer():
    
    
    
    def __init__(self, sed_df: pd.DataFrame, file_targ, source_lang="eng") -> None:
        self.df = sed_df
        self.file_targ = file_targ
        self.lang = source_lang
        
    def _voice_split(self):
        pass
        
    def _voice_recognize(self, onset: float, offset: float, callback=False):
       
        event_onset = onset
        event_duration = offset - onset
        
        print(f"Running voice recogniztion on {event_onset} to {offset}...")    
        
        
        y, sr = librosa.load(self.file_targ, sr=None, offset=event_onset, duration=event_duration)
        soundfile.write(file=TEMP_FOLDER_ABS+"/"+"stt_temp.wav", data=y, samplerate=sr, format="wav")

        
        event_text = get_by_ffmpeg.ffmpeg_sst("stt_temp.wav", lang=self.lang)
        
        return event_text
        
        
    def make_inference_result(self):
        
        text_all = []
        
        for (onset, offset) in zip(self.df.start, self.df.end):
            current_text = self._voice_recognize(onset=onset, offset=offset)
            text_all.append(current_text)
            
        df_with_text = self.df.copy(deep=True)
        df_with_text["recognized_text"] = text_all

        return df_with_text
            
        
def get_inference(file_targ, params_path, outputname, lang, post_process=True, output_folder="output", short_clip=0, device=None, inferer=None):
    resultout = None
    print("check cuda is available or not........")
    
    if torch.cuda.is_available():
        device = device if device else torch.device("cuda")
        ic(device)
    else:
        device = torch.device("cpu")
        ic(device)
        
    model = inferer if inferer else Pannscnn14attInferer
    print(f"Inferencing using model: {model.__name__}.\n")      
    
    outputname = outputname if outputname else "current"
    out_file = f"{output_folder}/{outputname}.csv"
    out_src_file = f"{output_folder}/{outputname}-all.csv"
    print("Checking file Type from file utils function check_type..\n")
    is_video = check_type(file_targ)
    ic(is_video)
    print("extracting audio from file utils.py..\n")
    targ_file_path = extract_audio(file_targ, format=DC.dataset_audio_format) if is_video else file_targ
    print("by using librosa library load the voice check channel")
    y, _ = mono_load(file_targ)
    
    
   
    print(f"using model to generate output...\n")    
    if short_clip:
        resultout = model(y, params_path, period=short_clip, device=device).get_breakpoint()
        print(f"Output breakpoint for short clip: {resultout}\n")
    else:
        resultout = model(y, params_path, device=device).make_inference_result()
        print(f"Output: {len(resultout)} breaks.\n")
        prediction_df = pd.DataFrame(resultout)
        output_df = prediction_df[prediction_df.speech_recognition=="speech"]
        print("Applying post process..\n")
       
        if post_process:
            output_df = SpeechSeries(output_df).series
            prediction_df = SpeechSeries(prediction_df).series
            print("Post process applied.\n")
        
        
        if lang: 
            output_df = SttInferer(output_df, file_targ=file_targ, source_lang=lang).make_inference_result()
            ic(output_df)
        
        
        output_df.to_csv(out_file, index=False)
        prediction_df.to_csv(out_src_file, index=False)
        
        print(f"Inference output file generated (This is not the final output), see: {output_folder}.\n")
    
    return output_df
