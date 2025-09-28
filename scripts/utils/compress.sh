filename=$1
# 1. Check if the file exists
if [ ! -f $filename ]; then
    echo "File not found!"
    exit 1
fi
tempname=$(basename $filename)_temp.mp4
ffmpeg -i $filename -c:v libx264 -crf 30 -preset veryfast -c:a aac -b:a 128k $tempname
mv $tempname $filename