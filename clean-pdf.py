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
import numpy as np
import cv2
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def clean_page(image, level=3, deskew=False):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if deskew:
        # Threshold to get text for deskewing
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45: angle = -(90 + angle)
            else: angle = -angle
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

def process_pdf(input_path, output_path, level=3, deskew=False, ocr=False, lang='eng', cover=False):
    print(f"Processing {input_path}...")
    
    if ocr:
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR not found. Please install it to use the --ocr flag.")
            print("On Debian/Ubuntu: sudo apt-get install tesseract-ocr")
            sys.exit(1)

    doc = fitz.open(input_path)
    output_doc = fitz.open()
    
    # Use a higher DPI for rendering (300 DPI is standard for high quality OCR/printing)
    zoom = 300 / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        print(f"  Page {page_num + 1}/{len(doc)}", end="\r")
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
        cleaned_img = clean_page(img, level=level, deskew=deskew)
        
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

    # Save with optimization and compression
    output_doc.save(output_path, garbage=3, deflate=True)
    output_doc.close()
    doc.close()
    print(f"\nDone! Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean scanned PDFs and remove yellow backgrounds.")
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output PDF file (default: input_cleaned.pdf)")
    parser.add_argument("-l", "--level", type=int, choices=range(1, 6), default=3, 
                        help="Aggressiveness level (1-5, default 3)")
    parser.add_argument("-d", "--deskew", action="store_true", help="Attempt to align/deskew pages")
    parser.add_argument("--ocr", action="store_true", help="Perform OCR to make the PDF searchable")
    parser.add_argument("--cover", action="store_true", help="Treat the first page as a cover (skip processing)")
    parser.add_argument("--lang", default="eng", help="OCR language (default: eng)")

    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base} - cleaned{ext}"

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        sys.exit(1)

    process_pdf(args.input, args.output, level=args.level, deskew=args.deskew, ocr=args.ocr, lang=args.lang, cover=args.cover)

if __name__ == "__main__":
    main()
