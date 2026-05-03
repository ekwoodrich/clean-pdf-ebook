#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pymupdf",
#   "pillow",
#   "opencv-python",
#   "numpy",
#   "pytesseract",
# ]
# ///
import argparse
import os
import sys
import shutil
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import subprocess
import tempfile

def clean_page(image, level=3, deskew=False):
# ... (rest of the clean_page function remains the same)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if deskew:
        # Threshold to get text/elements
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours to distinguish text from graphics
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_points = []
        has_large_graphics = False
        page_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Heuristic for text: relatively small area, not spanning whole page
            # Adjust these thresholds if needed
            if 10 < area < (page_area * 0.05) and w < (gray.shape[1] * 0.8):
                text_points.append(cnt)
            elif area > (page_area * 0.1):
                # If a single element takes up >10% of the page, it's likely a graphic
                has_large_graphics = True
        
        # Only deskew if we have enough text bits and no massive graphics to confuse it
        if len(text_points) > 20 and not has_large_graphics:
            all_text_coords = np.concatenate(text_points)
            angle = cv2.minAreaRect(all_text_coords)[-1]
            
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
            
            # Limit angle to avoid extreme rotations on noise
            if abs(angle) < 10 and abs(angle) > 0.1:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Denoise to remove small salt-and-pepper noise which can cause blobs
    denoised = cv2.medianBlur(gray, 3)

    # 2. Use Adaptive Thresholding instead of morphological division for better local contrast
    # This is more robust against uneven lighting (shadows near the spine, etc.)
    # block_size must be odd. Larger blocks = more context, better for large text.
    block_size = 31 + (level * 10) 
    if block_size % 2 == 0: block_size += 1
    
    # constant C is subtracted from the mean. 
    # Larger C = more aggressive background removal (whiter).
    C = 5 + (level * 3)
    
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, C
    )

    # 3. Final cleaning: use a bit of morphological opening to remove tiny specks 
    # if level is high, otherwise skip to preserve thin fonts.
    if level >= 4:
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Re-merge to BGR
    final_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return final_img

def process_pdf(input_path, output_path, level=3, deskew=False, ocr=False, ocrmypdf=False, lang='eng', cover=False):
    print(f"Processing {input_path}...")
    
    if ocr and ocrmypdf:
        print("Error: Cannot use both --ocr and --ocrmypdf. Please choose one.")
        sys.exit(1)

    if ocr:
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR not found. Please install it to use the --ocr flag.")
            print("On Debian/Ubuntu: sudo apt-get install tesseract-ocr")
            sys.exit(1)

    if ocrmypdf:
        # Check if ocrmypdf is available
        try:
            subprocess.run(["ocrmypdf", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: ocrmypdf not found. Please install it to use the --ocrmypdf flag.")
            sys.exit(1)

    doc = fitz.open(input_path)
    output_doc = fitz.open()
    
    # Use a higher DPI for rendering (300 DPI is standard for high quality OCR/printing)
    zoom = 300 / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        print(f"  Page {page_num + 1}/{len(doc)}")
        sys.stdout.flush()
        page = doc[page_num]
        
        # If --cover is set and it's the first page, copy it directly
        if cover and page_num == 0:
            output_doc.insert_pdf(doc, from_page=0, to_page=0)
            continue

        # Render page to image at high resolution
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert RGB to BGR for OpenCV
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Clean image
        # If using ocrmypdf, we disable our custom deskew to let ocrmypdf handle it better
        effective_deskew = deskew if not ocrmypdf else False
        cleaned_img = clean_page(img, level=level, deskew=effective_deskew)
        
        # Convert back to RGB/Grayscale for PIL
        # Since it's binary, 1-channel is enough and more efficient
        cleaned_gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(cleaned_gray)
        
        if ocr:
            # Perform OCR and get searchable PDF page
            # Tesseract handles high-res images well
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, extension='pdf', lang=lang)
            ocr_pdf = fitz.open("pdf", pdf_bytes)
            output_doc.insert_pdf(ocr_pdf)
        else:
            # Add the image as a new page with Flate compression (lossless)
            img_byte_arr = io.BytesIO()
            # Save as PNG to ensure lossless transfer of the binary image
            pil_img.save(img_byte_arr, format='PNG', optimize=True)
            img_bytes = img_byte_arr.getvalue()
            
            new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
            # insert_image will automatically use CCITT or Flate for binary/PNG
            new_page.insert_image(new_page.rect, stream=img_bytes)

    # Copy metadata from original to output
    output_doc.set_metadata(doc.metadata)

    if ocrmypdf:
        # Save to a temporary file first, then run ocrmypdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            output_doc.save(tmp_path, garbage=3, deflate=True)
            output_doc.close()
            
            print(f"Running ocrmypdf on cleaned pages...")
            cmd = ["ocrmypdf", "--language", lang, "--optimize", "1", "--skip-text", tmp_path, output_path]
            if deskew:
                cmd.append("--deskew")
            subprocess.run(cmd, check=True)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Save with optimization and compression
        output_doc.save(output_path, garbage=3, deflate=True)
        output_doc.close()
    
    doc.close()
    print(f"\nDone! Saved to {output_path}")

def process_file(input_path, output_path, args):
    _, ext = os.path.splitext(input_path.lower())
    
    # Ebook extensions to move directly
    EBOOK_EXTENSIONS = {'.epub', '.azw3', '.mobi', '.rtf', '.html', '.txt'}
    
    if ext == '.pdf':
        process_pdf(
            input_path, 
            output_path, 
            level=args.level, 
            deskew=args.deskew, 
            ocr=args.ocr, 
            ocrmypdf=args.ocrmypdf, 
            lang=args.lang, 
            cover=args.cover
        )
    elif ext in EBOOK_EXTENSIONS:
        print(f"Non-pdf ebook file found and moved without processing: {os.path.basename(input_path)}")
        shutil.move(input_path, output_path)
    else:
        # Silently ignore other extensions
        pass

def main():
    parser = argparse.ArgumentParser(description="Clean scanned PDFs and remove yellow backgrounds.")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output PDF file or directory")
    parser.add_argument("-l", "--level", type=int, choices=range(1, 6), default=3, 
                        help="Aggressiveness level (1-5, default 3)")
    parser.add_argument("-d", "--deskew", action="store_true", help="Attempt to align/deskew pages")
    parser.add_argument("--ocr", action="store_true", help="Perform OCR using pytesseract directly")
    parser.add_argument("--ocrmypdf", action="store_true", help="Perform OCR using ocrmypdf after cleaning")
    parser.add_argument("--cover", action="store_true", help="Treat the first page as a cover (skip processing)")
    parser.add_argument("--lang", default="eng", help="OCR language (default: eng)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Path {args.input} not found.")
        sys.exit(1)

    if os.path.isdir(args.input):
        # Batch processing mode
        input_dir = args.input
        output_dir = args.output if args.output else input_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            if os.path.isfile(input_path):
                base, ext = os.path.splitext(filename)
                if ext.lower() == '.pdf':
                    out_filename = f"{base} - cleaned{ext}"
                else:
                    out_filename = filename
                    
                output_path = os.path.join(output_dir, out_filename)
                process_file(input_path, output_path, args)
    else:
        # Single file mode
        input_path = args.input
        if not args.output:
            base, ext = os.path.splitext(input_path)
            if ext.lower() == '.pdf':
                output_path = f"{base} - cleaned{ext}"
            else:
                output_path = input_path # Should not really happen with the move logic but being safe
        else:
            output_path = args.output
            
        process_file(input_path, output_path, args)

if __name__ == "__main__":
    main()
