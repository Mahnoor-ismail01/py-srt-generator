# py-srt-generator..
A Python based Machine Learning Model that can generate SRT (subtitiles file) of any video efficiently.

## How to Run

First Clone the Repo
```ruby
git clone https://github.com/Mahnoor-ismail01/py-srt-generator.git
cd py-srt-generator
```


Next Install the required dependencies
```ruby

python -m pip install -r py-srt-generator/requirements.txt
```

Finally, Run the script under the project root folder
```ruby

python run.py
```

- Get your subtitle file under results folder under the project root folder.
- make the folder  **model**
- Download model from https://alphacephei.com/vosk/models
- place it under **model** folder.

- Then using Default model model_best.pth for generating subtitles. 
- Download model from https://drive.google.com/file/d/1SPwQ3tDTyUQEfAwLNQungZwXhYxA5E_I/view
- Extract this model and place it under **models** folder.


-Change the default model path you want to use in config.py INFERENCE_PARAMS_PATH under the project root folder.


Enjoy  :)


