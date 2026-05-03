#!/bin/bash
set -e

# Defaults for environment variables if not set
INPUT_DIR=${INPUT_DIR:-/input}
OUTPUT_DIR=${OUTPUT_DIR:-/output}
ARCHIVE_DIR=${ARCHIVE_DIR:-/archive}
FAILED_DIR=${FAILED_DIR:-/failed}
FLAGS=${FLAGS:-"-l 3"}

# Find all PDFs in the input directory
shopt -s nullglob
INPUT_FILES=("$INPUT_DIR"/*.pdf)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "No PDF files found in $INPUT_DIR"
    exit 0
fi

echo "Found ${#INPUT_FILES[@]} PDF(s) to process."

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    FILENAME=$(basename "$INPUT_FILE")
    BASENAME="${FILENAME%.*}"
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME - cleaned.pdf"

    echo "--------------------------------------------"
    echo "Processing $FILENAME..."
    echo "Flags: $FLAGS"

    # Run the cleaning script using uv
    if uv run clean_pdf.py "$INPUT_FILE" -o "$OUTPUT_FILE" $FLAGS; then
        echo "Cleaning complete. Output saved to $OUTPUT_FILE"
        
        # Move the original file to the archive directory
        echo "Archiving $FILENAME to $ARCHIVE_DIR..."
        mv "$INPUT_FILE" "$ARCHIVE_DIR/"
    else
        echo "ERROR: Failed to process $FILENAME. Moving to $FAILED_DIR to prevent infinite loops."
        mv "$INPUT_FILE" "$FAILED_DIR/"
    fi
done

echo "--------------------------------------------"
echo "Batch processing complete."
