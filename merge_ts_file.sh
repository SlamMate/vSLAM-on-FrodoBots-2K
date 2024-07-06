#!/bin/bash

# Change to the directory containing the TS files
cd ./recordings

# Create file list
# Find all files matching the pattern and generate the file list for FFmpeg
find . -type f -name '*uid_s_1000__uid_e_video_*.ts' \
    | sort \
    | awk '{print "file \x27" $0 "\x27"}' > filelist.txt

# Merge TS files
ffmpeg -f concat -safe 0 -i filelist.txt -c copy rgb.ts


# Clean up the file list
rm filelist.txt
