#!/bin/bash

# List of input videos
videos=("video_input.mp4" "video_input1.mp4" "video_input2.mp4" "video_input3.mp4")

# Loop through each video and extract I, P, and B frames
for video in "${videos[@]}"; do
    # Get base name without extension
    base="${video%.*}"

    echo "Processing $video ..."

    # Extract I-frames
    ffmpeg -y -i "$video" -vf "select=eq(pict_type\,I),setpts=N/FRAME_RATE/TB" -an "${base}_Iframes.mp4"

    # Extract P-frames
    ffmpeg -y -i "$video" -vf "select=eq(pict_type\,P),setpts=N/FRAME_RATE/TB" -an "${base}_Pframes.mp4"

    # Extract B-frames
    ffmpeg -y -i "$video" -vf "select=eq(pict_type\,B),setpts=N/FRAME_RATE/TB" -an "${base}_Bframes.mp4"

    echo "Finished processing $video"
done
