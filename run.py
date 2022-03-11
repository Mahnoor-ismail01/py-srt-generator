
import sys

import pandas as pd
import logging
from inferer import get_inference
from encoder import  SRTEncoder
from config import INFERENCE_PARAMS_PATH, OUTPUT_FOLDER_ABS
import os

def get_logfilename()->str:
    DIR = os.path.dirname(os.path.abspath(__file__))
    return f"{DIR}/logs.log"



logging.basicConfig(
        filename=get_logfilename(),
        filemode="a",
        format="%(asctime)s - [%(levelname)s] -  %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - "
               "%(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.DEBUG,
    )




def SRTgen(name_of_target_file, outputname, sub_format: str="ass", lang: str="eng", post=True, output_folder=OUTPUT_FOLDER_ABS):
    print("call inferer.py method get_inference ")
    get_inference(file_targ=name_of_target_file,
                params_path=INFERENCE_PARAMS_PATH,
                outputname=outputname,
                lang=lang,
                post_process=post,
                output_folder=output_folder)

    df = pd.read_csv(f"{output_folder}/{outputname}.csv")

    if ("srt" in outputname) or (sub_format=="srt") or (sub_format==".srt"):
        encoder = SRTEncoder(df)
        
    print(f"Calling encoder to generate the final output...\n") 
    encoder.generate(outputname)
    print(f"All procedures done! Subtitle file generated.\n")


    
    
    print("The output subtitle file is under project folder..\n\n")
        
def main(input_file=None, output_file=None):
    # DIR
    print("****SUBTITLE GENERATION***\n")

    file=open("logsmy.txt","w")
    file.write("")
    file.close()
    
    if not input_file:
        name_file = input("full path for file :\n\n")
    else:
        name_file = input_file
    
    
    if not output_file:
        
        file_name_output = input("output file name:")
        file_name_output = "current" if not file_name_output else file_name_output
    else:
        file_name_output = output_file

    

       
    languagedict = {'0':'eng'}  
    index_of_lang = '0'
    language = languagedict[index_of_lang ] if index_of_lang .strip() else ""   
    
    
    x = SRTgen(name_of_target_file=name_file, outputname=file_name_output, lang=language)


if __name__ == '__main__':
    if sys.argv[1:]:
        print(sys.argv)
        main(sys.argv[1], sys.argv[2])
    else:
        main()




    