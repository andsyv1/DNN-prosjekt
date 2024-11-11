#!/bin/bash

# Enable nullglob in case no files match the pattern
shopt -s nullglob

# Array of files starting with "mask_" and ending with ".txt"
files=(mask_*.txt)

# Check if any files were found
if [ "${#files[@]}" -eq 0 ]; then
    echo "No files found starting with 'mask_'."
    exit 1
fi

# Loop through each file and rename it
for file in "${files[@]}"; do
    # Remove 'mask_' prefix from the filename
    base="${file#mask_}"
    # Construct the new filename with 'fish_' prefix
    newfile="fish_$base"
    echo "Renaming '$file' to '$newfile'"
    mv -- "$file" "$newfile"
done
