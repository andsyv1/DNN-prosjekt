#!/bin/bash

# Loop through numbers 01 to 23
for i in $(seq -w 1 23); do
    dir="polygon_$i"
    if [ -d "$dir" ]; then
        echo "Processing directory $dir"
        # Copy all contents, including hidden files, to the current directory
        cp -a "$dir"/. .
        # Remove the subdirectory
        rm -rf "$dir"
    else
        echo "Directory $dir does not exist."
    fi
done
