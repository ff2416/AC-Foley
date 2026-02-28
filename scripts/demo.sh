#!/bin/bash

# video content matching the audio 
python inf.py \
--video ./examples/video/lion.mp4 \
--audio ./examples/audio/lion.mp3 \
--prompt "lion roaring"

# video not content matching the audio
python inf.py \
--video ./examples/video/lion.mp4 \
--audio ./examples/audio/cat.wav \
--prompt "cat meowing" \
--mask_away_clip

# video not content matching the audio
python inf.py \
--video ./examples/video/lion.mp4 \
--audio ./examples/audio/cow.wav \
--prompt "cow mooing" \
--mask_away_clip

