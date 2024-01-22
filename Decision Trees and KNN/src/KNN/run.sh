#!/bin/bash
# this code is referred from chat-gpt
# Check if a filename argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Extract the filename argument
filename="$1"

if [ -e "$filename" ]; then
    echo "File '$filename' exists."
    # Check if the file is a regular file (not a directory or special file)
    if [ -f "$filename" ]; then
        echo "File '$filename' is a regular file."
    else
        echo "File '$filename' is not a regular file."
        exit 1
    fi
else
    echo "File '$filename' does not exist or is invalid."
    exit 1
fi

# Run the Python script with the provided filename as an argument
python run_KNN.py "$filename"
