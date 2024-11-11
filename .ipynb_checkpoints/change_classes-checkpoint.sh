#!/bin/bash

# Check if the directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Directory containing files
DIR="$1"

# Process each file in the directory
for FILE in "$DIR"/*; do
  if [ -f "$FILE" ]; then
    # Create a temporary file to store modified content
    TEMP_FILE=$(mktemp)
    
    # Read each line in the file
    while IFS= read -r LINE || [ -n "$LINE" ]; do
      # Decrement the first number on the line
      MODIFIED_LINE=$(echo "$LINE" | sed -E 's/^([0-9]+)/echo $((\1 - 1))/e')
      echo "$MODIFIED_LINE" >> "$TEMP_FILE"
    done < "$FILE"
    
    # Replace the original file with the modified content
    mv "$TEMP_FILE" "$FILE"
  fi
done

echo "Processing complete."
