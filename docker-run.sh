#!/bin/bash
set -e

# Defaults for environment variables if not set
INPUT_DIR=${INPUT_DIR:-/input}
OUTPUT_DIR=${OUTPUT_DIR:-/output}
ARCHIVE_DIR=${ARCHIVE_DIR:-/archive}
FLAGS=${FLAGS:-"-l 3"}

# Find the first PDF in the input directory
INPUT_FILE=$(ls "$INPUT_DIR"/*.pdf | head -n 1)

if [ -z "$INPUT_FILE" ]; then
    echo "No PDF files found in $INPUT_DIR"
    exit 0
fi

FILENAME=$(basename "$INPUT_FILE")
BASENAME="${FILENAME%.*}"
OUTPUT_FILE="$OUTPUT_DIR/$BASENAME - cleaned.pdf"

echo "Processing $FILENAME..."
echo "Flags: $FLAGS"

# Run the cleaning script using uv
uv run clean_pdf.py "$INPUT_FILE" -o "$OUTPUT_FILE" $FLAGS

echo "Cleaning complete. Output saved to $OUTPUT_FILE"

# Move the original file to the archive directory
echo "Archiving $FILENAME to $ARCHIVE_DIR..."
mv "$INPUT_FILE" "$ARCHIVE_DIR/"

echo "Done."
