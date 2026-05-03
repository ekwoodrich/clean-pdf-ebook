#!/bin/bash
set -e

# Defaults for environment variables if not set
INPUT_DIR=${INPUT_DIR:-/input}
OUTPUT_DIR=${OUTPUT_DIR:-/output}
ARCHIVE_DIR=${ARCHIVE_DIR:-/archive}
FAILED_DIR=${FAILED_DIR:-/failed}
FLAGS=${FLAGS:-"-l 3"}

# Find all supported files in the input directory
shopt -s nullglob
INPUT_FILES=("$INPUT_DIR"/*.pdf "$INPUT_DIR"/*.epub "$INPUT_DIR"/*.azw3 "$INPUT_DIR"/*.mobi "$INPUT_DIR"/*.rtf "$INPUT_DIR"/*.html "$INPUT_DIR"/*.txt)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "No supported ebook files found in $INPUT_DIR"
    exit 0
fi

echo "Found ${#INPUT_FILES[@]} file(s) to process."

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    FILENAME=$(basename "$INPUT_FILE")
    EXTENSION="${FILENAME##*.}"
    BASENAME="${FILENAME%.*}"
    
    if [ "$EXTENSION" == "pdf" ]; then
        OUTPUT_FILE="$OUTPUT_DIR/$BASENAME - cleaned.pdf"
    else
        OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
    fi

    echo "--------------------------------------------"
    echo "Processing $FILENAME..."
    
    # Run the cleaning script using uv
    # We use a temporary log to see if the file was moved by the script
    if uv run clean_pdf.py "$INPUT_FILE" -o "$OUTPUT_FILE" $FLAGS; then
        # If the file still exists in INPUT_DIR, it means it was a PDF that was processed
        # or an ebook that failed to move (unlikely if exit code was 0).
        # If the file is GONE, it was moved by clean_pdf.py (ebook logic).
        if [ -f "$INPUT_FILE" ]; then
            echo "Cleaning complete. Output saved to $OUTPUT_FILE"
            echo "Archiving $FILENAME to $ARCHIVE_DIR..."
            mv "$INPUT_FILE" "$ARCHIVE_DIR/"
        else
            echo "File $FILENAME was handled and moved to output."
        fi
    else
        echo "ERROR: Failed to process $FILENAME. Moving to $FAILED_DIR to prevent infinite loops."
        if [ -f "$INPUT_FILE" ]; then
            mv "$INPUT_FILE" "$FAILED_DIR/"
        fi
    fi
done

echo "--------------------------------------------"
echo "Batch processing complete."
