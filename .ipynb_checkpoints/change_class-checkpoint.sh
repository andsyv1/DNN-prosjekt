#!/bin/bash

# Enable nullglob in case no files match the pattern
shopt -s nullglob

# Loop through all files in the current directory
for file in *; do
    if [ -f "$file" ]; then
        echo "Processing '$file'"
        # Replace the first number on each line with 0
        if sed --version >/dev/null 2>&1; then
            # GNU sed syntax
            sed -i 's/[0-9]\+/0/' "$file"
        else
            # BSD/macOS sed syntax
            sed -i '' 's/[0-9][0-9]*/0/' "$file"
        fi
    fi
done

